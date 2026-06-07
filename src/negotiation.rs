//! Field negotiation protocol: agents propose direction changes,
//! the field mediates conflicts, and weighted consensus emerges.

use crate::field::{IntentionField, IntentionVector};
use crate::agent_contribution::ContributionAggregator;

/// A proposal from an agent to change the field direction.
#[derive(Debug, Clone)]
pub struct DirectionProposal {
    pub agent_id: usize,
    pub proposed_intention: IntentionVector,
    pub confidence: f64,
    pub reason: ProposalReason,
}

/// Why the agent is proposing a change.
#[derive(Debug, Clone, PartialEq)]
pub enum ProposalReason {
    /// Agent discovered something musically interesting.
    Exploration,
    /// Agent believes the current direction is stale.
    Staleness,
    /// Agent wants to return to a known good state.
    ReturnToCenter,
    /// Agent received external input (e.g., audience cue).
    ExternalCue,
}

/// Outcome of a negotiation round.
#[derive(Debug, Clone, PartialEq)]
pub enum NegotiationOutcome {
    /// Proposal accepted; field shifts toward the proposed direction.
    Accepted { shift: IntentionVector },
    /// Proposal rejected; field stays the same.
    Rejected,
    /// Compromise reached; field shifts partially.
    Compromise { shift: IntentionVector, support: f64 },
    /// Conflict; no consensus, field holds steady.
    Conflict,
}

/// Configuration for the negotiation protocol.
#[derive(Debug, Clone)]
pub struct NegotiationConfig {
    /// Minimum support fraction to accept a proposal (0.0–1.0).
    pub acceptance_threshold: f64,
    /// Minimum support to even consider a compromise.
    pub compromise_threshold: f64,
    /// How much weight the current field state has vs. proposals.
    pub field_inertia: f64,
    /// Maximum allowed shift magnitude per negotiation round.
    pub max_shift: f64,
}

impl Default for NegotiationConfig {
    fn default() -> Self {
        Self {
            acceptance_threshold: 0.6,
            compromise_threshold: 0.3,
            field_inertia: 0.5,
            max_shift: 2.0,
        }
    }
}

/// The negotiation mediator that resolves proposals.
pub struct FieldNegotiator {
    pub config: NegotiationConfig,
}

impl FieldNegotiator {
    pub fn new(config: NegotiationConfig) -> Self {
        Self { config }
    }

    pub fn default_negotiator() -> Self {
        Self::new(NegotiationConfig::default())
    }

    /// Evaluate a single proposal against the current field and agent contributions.
    pub fn evaluate_proposal(
        &self,
        proposal: &DirectionProposal,
        field: &IntentionField,
        aggregator: &ContributionAggregator,
    ) -> NegotiationOutcome {
        if aggregator.is_empty() {
            return NegotiationOutcome::Accepted {
                shift: proposal.proposed_intention.add(&field.state.scale(-1.0)),
            };
        }

        // Count supporting vs opposing agents
        let mut support_weight = 0.0;
        let mut oppose_weight = 0.0;
        let mut compromise_direction = IntentionVector::zero(field.dimensions);
        let total_weight: f64 = aggregator.contributions.iter().map(|c| c.weight).sum();

        if total_weight < 1e-10 {
            return NegotiationOutcome::Rejected;
        }

        for contribution in &aggregator.contributions {
            let alignment = contribution.intention.dot(&proposal.proposed_intention)
                / (contribution.intention.norm() * proposal.proposed_intention.norm().max(1e-10));
            if alignment > 0.0 {
                support_weight += contribution.weight;
                compromise_direction = compromise_direction.add(
                    &proposal.proposed_intention.scale(contribution.weight)
                        .add(&contribution.intention.scale(contribution.weight))
                        .scale(0.5),
                );
            } else {
                oppose_weight += contribution.weight;
                compromise_direction = compromise_direction.add(&contribution.intention.scale(contribution.weight));
            }
        }

        let support_fraction = support_weight / total_weight;

        // Compute proposed shift
        let raw_shift = proposal.proposed_intention.add(&field.state.scale(-1.0));
        let shift_magnitude = raw_shift.norm();

        // Clamp shift
        let shift = if shift_magnitude > self.config.max_shift {
            raw_shift.scale(self.config.max_shift / shift_magnitude)
        } else {
            raw_shift
        };

        if support_fraction >= self.config.acceptance_threshold {
            NegotiationOutcome::Accepted { shift }
        } else if support_fraction >= self.config.compromise_threshold {
            let compromise_norm = compromise_direction.norm();
            let clamped_compromise = if compromise_norm > self.config.max_shift {
                compromise_direction.scale(self.config.max_shift / compromise_norm)
            } else if compromise_norm < 1e-10 {
                IntentionVector::zero(field.dimensions)
            } else {
                compromise_direction
            };
            NegotiationOutcome::Compromise {
                shift: clamped_compromise,
                support: support_fraction,
            }
        } else if oppose_weight > support_weight {
            NegotiationOutcome::Conflict
        } else {
            NegotiationOutcome::Rejected
        }
    }

    /// Apply a negotiation outcome to the field.
    pub fn apply_outcome(&self, outcome: &NegotiationOutcome, field: &mut IntentionField) {
        let shift = match outcome {
            NegotiationOutcome::Accepted { shift } => shift,
            NegotiationOutcome::Compromise { shift, .. } => shift,
            _ => return,
        };

        let inertia_shift = shift.scale(1.0 - self.config.field_inertia);
        let new_state = field.state.add(&inertia_shift);
        field.update(new_state, field.n_agents);
    }

    /// Run a multi-proposal negotiation round.
    /// Returns outcomes in the same order as proposals.
    pub fn negotiate_round(
        &self,
        proposals: &[DirectionProposal],
        field: &IntentionField,
        aggregator: &ContributionAggregator,
    ) -> Vec<NegotiationOutcome> {
        proposals
            .iter()
            .map(|p| self.evaluate_proposal(p, field, aggregator))
            .collect()
    }

    /// Find the best accepted proposal from a round.
    pub fn best_accepted(
        outcomes: &[(DirectionProposal, NegotiationOutcome)],
    ) -> Option<&NegotiationOutcome> {
        outcomes.iter().find_map(|(_, o)| match o {
            NegotiationOutcome::Accepted { .. } => Some(o),
            _ => None,
        })
    }

    /// Count outcomes by type.
    pub fn count_by_type(outcomes: &[NegotiationOutcome]) -> (usize, usize, usize, usize) {
        let mut accepted = 0;
        let mut rejected = 0;
        let mut compromise = 0;
        let mut conflict = 0;
        for o in outcomes {
            match o {
                NegotiationOutcome::Accepted { .. } => accepted += 1,
                NegotiationOutcome::Rejected => rejected += 1,
                NegotiationOutcome::Compromise { .. } => compromise += 1,
                NegotiationOutcome::Conflict => conflict += 1,
            }
        }
        (accepted, rejected, compromise, conflict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_contribution::AgentContribution;
    use crate::field::IntentionVector;

    fn make_field() -> IntentionField {
        let mut f = IntentionField::new(2);
        f.update(IntentionVector::new(vec![1.0, 0.0]), 3);
        f
    }

    fn make_aligned_aggregator() -> ContributionAggregator {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(1, IntentionVector::new(vec![0.9, 0.1]), 1.0));
        agg.add(AgentContribution::new(2, IntentionVector::new(vec![1.0, 0.1]), 1.0));
        agg
    }

    #[test]
    fn test_empty_aggregator_accepts() {
        let negotiator = FieldNegotiator::default_negotiator();
        let field = make_field();
        let agg = ContributionAggregator::new();
        let proposal = DirectionProposal {
            agent_id: 0,
            proposed_intention: IntentionVector::new(vec![2.0, 0.0]),
            confidence: 1.0,
            reason: ProposalReason::Exploration,
        };
        let outcome = negotiator.evaluate_proposal(&proposal, &field, &agg);
        assert!(matches!(outcome, NegotiationOutcome::Accepted { .. }));
    }

    #[test]
    fn test_aligned_proposal_accepted() {
        let negotiator = FieldNegotiator::default_negotiator();
        let field = make_field();
        let agg = make_aligned_aggregator();
        let proposal = DirectionProposal {
            agent_id: 0,
            proposed_intention: IntentionVector::new(vec![1.5, 0.0]),
            confidence: 1.0,
            reason: ProposalReason::Exploration,
        };
        let outcome = negotiator.evaluate_proposal(&proposal, &field, &agg);
        assert!(matches!(outcome, NegotiationOutcome::Accepted { .. }));
    }

    #[test]
    fn test_opposing_proposal_rejected() {
        let negotiator = FieldNegotiator::default_negotiator();
        let field = make_field();
        let agg = make_aligned_aggregator();
        let proposal = DirectionProposal {
            agent_id: 0,
            proposed_intention: IntentionVector::new(vec![-5.0, 0.0]),
            confidence: 1.0,
            reason: ProposalReason::Exploration,
        };
        let outcome = negotiator.evaluate_proposal(&proposal, &field, &agg);
        assert!(matches!(outcome, NegotiationOutcome::Conflict | NegotiationOutcome::Rejected));
    }

    #[test]
    fn test_apply_outcome_accepted() {
        let negotiator = FieldNegotiator::default_negotiator();
        let mut field = make_field();
        let shift = IntentionVector::new(vec![1.0, 0.0]);
        let outcome = NegotiationOutcome::Accepted { shift };
        negotiator.apply_outcome(&outcome, &mut field);
        // With inertia 0.5, shift applied is 0.5 * (1.0, 0.0), so new state = (1.5, 0.0)
        assert!((field.state.dims[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_outcome_rejected() {
        let negotiator = FieldNegotiator::default_negotiator();
        let mut field = make_field();
        let original = field.state.clone();
        negotiator.apply_outcome(&NegotiationOutcome::Rejected, &mut field);
        assert_eq!(field.state, original);
    }

    #[test]
    fn test_negotiate_round() {
        let negotiator = FieldNegotiator::default_negotiator();
        let field = make_field();
        let agg = make_aligned_aggregator();
        let proposals = vec![
            DirectionProposal {
                agent_id: 0,
                proposed_intention: IntentionVector::new(vec![1.5, 0.0]),
                confidence: 1.0,
                reason: ProposalReason::Exploration,
            },
            DirectionProposal {
                agent_id: 1,
                proposed_intention: IntentionVector::new(vec![-5.0, 0.0]),
                confidence: 0.5,
                reason: ProposalReason::ReturnToCenter,
            },
        ];
        let outcomes = negotiator.negotiate_round(&proposals, &field, &agg);
        assert_eq!(outcomes.len(), 2);
    }

    #[test]
    fn test_count_by_type() {
        let outcomes = vec![
            NegotiationOutcome::Accepted { shift: IntentionVector::new(vec![1.0]) },
            NegotiationOutcome::Rejected,
            NegotiationOutcome::Compromise { shift: IntentionVector::new(vec![1.0]), support: 0.4 },
            NegotiationOutcome::Conflict,
        ];
        let (a, r, c, cf) = FieldNegotiator::count_by_type(&outcomes);
        assert_eq!((a, r, c, cf), (1, 1, 1, 1));
    }

    #[test]
    fn test_best_accepted() {
        let outcomes = vec![
            (
                DirectionProposal {
                    agent_id: 0,
                    proposed_intention: IntentionVector::new(vec![1.0]),
                    confidence: 1.0,
                    reason: ProposalReason::Exploration,
                },
                NegotiationOutcome::Rejected,
            ),
            (
                DirectionProposal {
                    agent_id: 1,
                    proposed_intention: IntentionVector::new(vec![2.0]),
                    confidence: 1.0,
                    reason: ProposalReason::ExternalCue,
                },
                NegotiationOutcome::Accepted { shift: IntentionVector::new(vec![1.0]) },
            ),
        ];
        let best = FieldNegotiator::best_accepted(&outcomes);
        assert!(best.is_some());
    }

    #[test]
    fn test_shift_clamped() {
        let config = NegotiationConfig {
            max_shift: 0.5,
            ..Default::default()
        };
        let negotiator = FieldNegotiator::new(config);
        let field = make_field();
        let agg = make_aligned_aggregator();
        let proposal = DirectionProposal {
            agent_id: 0,
            proposed_intention: IntentionVector::new(vec![100.0, 0.0]),
            confidence: 1.0,
            reason: ProposalReason::Exploration,
        };
        let outcome = negotiator.evaluate_proposal(&proposal, &field, &agg);
        if let NegotiationOutcome::Accepted { shift } = &outcome {
            assert!(shift.norm() <= 0.5 + 1e-10, "shift should be clamped");
        }
    }

    #[test]
    fn test_apply_compromise() {
        let config = NegotiationConfig {
            field_inertia: 0.0,
            ..Default::default()
        };
        let negotiator = FieldNegotiator::new(config);
        let mut field = make_field();
        let shift = IntentionVector::new(vec![0.0, 1.0]);
        let outcome = NegotiationOutcome::Compromise { shift: shift.clone(), support: 0.4 };
        negotiator.apply_outcome(&outcome, &mut field);
        assert!((field.state.dims[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conflict_does_not_modify() {
        let negotiator = FieldNegotiator::default_negotiator();
        let mut field = make_field();
        let original = field.state.clone();
        negotiator.apply_outcome(&NegotiationOutcome::Conflict, &mut field);
        assert_eq!(field.state, original);
    }
}
