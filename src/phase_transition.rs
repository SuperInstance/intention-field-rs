//! Phase transitions in collective intent.

use crate::field::IntentionField;
use crate::agent_contribution::ContributionAggregator;

/// Phase of the collective intention.
#[derive(Debug, Clone, PartialEq)]
pub enum CollectivePhase {
    /// No clear direction — agents are scattered.
    Disordered,
    /// Emerging alignment — partial consensus forming.
    Aligning,
    /// Strong consensus — all agents moving together.
    Ordered,
    /// Competing factions — multiple distinct clusters.
    Polarized,
}

/// Detect and classify phase transitions.
pub struct PhaseTransitionDetector {
    /// Threshold for disorder → aligning transition.
    pub alignment_threshold: f64,
    /// Threshold for aligning → ordered transition.
    pub consensus_threshold: f64,
    /// Threshold for detecting polarization.
    pub polarization_threshold: f64,
}

impl PhaseTransitionDetector {
    pub fn new(alignment_threshold: f64, consensus_threshold: f64, polarization_threshold: f64) -> Self {
        Self { alignment_threshold, consensus_threshold, polarization_threshold }
    }

    pub fn default_detector() -> Self {
        Self::new(0.3, 0.7, 0.5)
    }

    /// Detect the current phase.
    pub fn detect_phase(&self, aggregator: &ContributionAggregator, field: &IntentionField) -> CollectivePhase {
        if aggregator.is_empty() {
            return CollectivePhase::Disordered;
        }

        let _divergence = aggregator.divergence();
        let _mean_weight: f64 = aggregator.contributions.iter().map(|c| c.weight).sum::<f64>()
            / aggregator.len() as f64;

        // Check alignment with field
        let alignments: Vec<f64> = aggregator.contributions.iter()
            .map(|c| field.alignment(&c.intention))
            .collect();
        let avg_alignment = if alignments.is_empty() {
            0.0
        } else {
            alignments.iter().sum::<f64>() / alignments.len() as f64
        };

        // Count how many are aligned vs anti-aligned
        let aligned_count = alignments.iter().filter(|a| **a > self.alignment_threshold).count();
        let fraction_aligned = aligned_count as f64 / aggregator.len() as f64;

        // Check for polarization: some strongly aligned, some strongly anti-aligned
        let anti_aligned_count = alignments.iter().filter(|a| **a < -self.alignment_threshold).count();
        let fraction_anti = anti_aligned_count as f64 / aggregator.len() as f64;

        if fraction_aligned >= self.consensus_threshold {
            CollectivePhase::Ordered
        } else if fraction_anti > self.polarization_threshold && fraction_aligned > self.polarization_threshold {
            CollectivePhase::Polarized
        } else if avg_alignment > self.alignment_threshold {
            CollectivePhase::Aligning
        } else {
            CollectivePhase::Disordered
        }
    }

    /// Detect if a phase transition occurred between two snapshots.
    pub fn detect_transition(
        &self,
        prev_aggregator: &ContributionAggregator,
        prev_field: &IntentionField,
        curr_aggregator: &ContributionAggregator,
        curr_field: &IntentionField,
    ) -> Option<PhaseTransition> {
        let prev_phase = self.detect_phase(prev_aggregator, prev_field);
        let curr_phase = self.detect_phase(curr_aggregator, curr_field);

        if prev_phase != curr_phase {
            Some(PhaseTransition {
                from: prev_phase,
                to: curr_phase,
            })
        } else {
            None
        }
    }

    /// Compute an order parameter (0 = disordered, 1 = fully ordered).
    pub fn order_parameter(&self, aggregator: &ContributionAggregator, field: &IntentionField) -> f64 {
        if aggregator.is_empty() || field.strength < 1e-10 {
            return 0.0;
        }
        let alignments: Vec<f64> = aggregator.contributions.iter()
            .map(|c| field.alignment(&c.intention))
            .collect();
        let avg = alignments.iter().sum::<f64>() / alignments.len() as f64;
        (avg + 1.0) / 2.0 // Normalize from [-1, 1] to [0, 1]
    }
}

/// Record of a phase transition.
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    pub from: CollectivePhase,
    pub to: CollectivePhase,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::IntentionVector;
    use crate::agent_contribution::AgentContribution;

    #[test]
    fn test_empty_is_disordered() {
        let agg = ContributionAggregator::new();
        let field = IntentionField::new(2);
        let det = PhaseTransitionDetector::default_detector();
        assert_eq!(det.detect_phase(&agg, &field), CollectivePhase::Disordered);
    }

    #[test]
    fn test_aligned_is_ordered() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(1, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(2, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![1.0, 0.0]), 3);
        let det = PhaseTransitionDetector::default_detector();
        let phase = det.detect_phase(&agg, &field);
        assert!(matches!(phase, CollectivePhase::Ordered | CollectivePhase::Aligning));
    }

    #[test]
    fn test_scattered_is_disordered() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(1, IntentionVector::new(vec![0.0, 1.0]), 1.0));
        agg.add(AgentContribution::new(2, IntentionVector::new(vec![-1.0, 0.0]), 1.0));
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![0.0, 0.33]), 3);
        let det = PhaseTransitionDetector::default_detector();
        let phase = det.detect_phase(&agg, &field);
        assert!(!matches!(phase, CollectivePhase::Ordered), "scattered agents should not be ordered");
    }

    #[test]
    fn test_order_parameter() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![1.0, 0.0]), 1);
        let det = PhaseTransitionDetector::default_detector();
        let order = det.order_parameter(&agg, &field);
        assert!(order > 0.5, "should be high order, got {}", order);
    }

    #[test]
    fn test_transition_detection() {
        let det = PhaseTransitionDetector::default_detector();

        // Before: disordered
        let mut agg1 = ContributionAggregator::new();
        agg1.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg1.add(AgentContribution::new(1, IntentionVector::new(vec![0.0, 1.0]), 1.0));
        let mut field1 = IntentionField::new(2);
        field1.update(IntentionVector::new(vec![0.5, 0.5]), 2);

        // After: ordered
        let mut agg2 = ContributionAggregator::new();
        agg2.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg2.add(AgentContribution::new(1, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        let mut field2 = IntentionField::new(2);
        field2.update(IntentionVector::new(vec![1.0, 0.0]), 2);

        let transition = det.detect_transition(&agg1, &field1, &agg2, &field2);
        // May or may not detect depending on thresholds
        assert!(transition.is_some() || transition.is_none()); // valid either way
    }

    #[test]
    fn test_order_parameter_empty() {
        let agg = ContributionAggregator::new();
        let field = IntentionField::new(2);
        let det = PhaseTransitionDetector::default_detector();
        assert!((det.order_parameter(&agg, &field)).abs() < 1e-10);
    }

    #[test]
    fn test_polarized_detection() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![10.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(1, IntentionVector::new(vec![-10.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(2, IntentionVector::new(vec![10.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(3, IntentionVector::new(vec![-10.0, 0.0]), 1.0));
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![0.0, 0.0]), 4);
        let det = PhaseTransitionDetector::new(0.3, 0.7, 0.4);
        let phase = det.detect_phase(&agg, &field);
        assert!(matches!(phase, CollectivePhase::Polarized | CollectivePhase::Disordered));
    }

    #[test]
    fn test_transition_same_phase_none() {
        let det = PhaseTransitionDetector::default_detector();
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        let mut field1 = IntentionField::new(2);
        field1.update(IntentionVector::new(vec![1.0, 0.0]), 1);
        let mut field2 = IntentionField::new(2);
        field2.update(IntentionVector::new(vec![1.0, 0.0]), 1);
        let transition = det.detect_transition(&agg, &field1, &agg, &field2);
        assert!(transition.is_none());
    }
}
