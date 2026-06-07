//! Agent contributions to the intention field.

use crate::field::{IntentionField, IntentionVector};

/// A single agent's contribution to the shared field.
#[derive(Debug, Clone)]
pub struct AgentContribution {
    pub agent_id: usize,
    pub intention: IntentionVector,
    /// Confidence/weight of this contribution.
    pub weight: f64,
}

impl AgentContribution {
    pub fn new(agent_id: usize, intention: IntentionVector, weight: f64) -> Self {
        Self { agent_id, intention, weight }
    }

    /// Weighted intention vector.
    pub fn weighted(&self) -> IntentionVector {
        self.intention.scale(self.weight)
    }
}

/// Aggregator for combining agent contributions.
#[derive(Debug, Clone)]
pub struct ContributionAggregator {
    pub contributions: Vec<AgentContribution>,
}

impl Default for ContributionAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl ContributionAggregator {
    pub fn new() -> Self {
        Self { contributions: Vec::new() }
    }

    /// Add a contribution.
    pub fn add(&mut self, contribution: AgentContribution) {
        // Replace existing contribution from same agent
        if let Some(existing) = self.contributions.iter_mut().find(|c| c.agent_id == contribution.agent_id) {
            *existing = contribution;
        } else {
            self.contributions.push(contribution);
        }
    }

    /// Remove an agent's contribution.
    pub fn remove(&mut self, agent_id: usize) {
        self.contributions.retain(|c| c.agent_id != agent_id);
    }

    /// Aggregate using weighted average.
    pub fn aggregate_weighted_average(&self) -> IntentionVector {
        if self.contributions.is_empty() {
            return IntentionVector::zero(0);
        }
        let dims = self.contributions[0].intention.dimension();
        let mut result = vec![0.0; dims];
        let total_weight: f64 = self.contributions.iter().map(|c| c.weight).sum();

        if total_weight < 1e-10 {
            return IntentionVector::zero(dims);
        }

        for c in &self.contributions {
            for (i, val) in c.intention.dims.iter().enumerate() {
                result[i] += val * c.weight / total_weight;
            }
        }

        IntentionVector::new(result)
    }

    /// Aggregate using consensus (find closest intention to all).
    pub fn aggregate_consensus(&self) -> IntentionVector {
        // Simplified: use the contribution with highest weight
        self.contributions.iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.intention.clone())
            .unwrap_or_else(|| IntentionVector::zero(0))
    }

    /// Apply aggregated contributions to a field.
    pub fn apply_to_field(&self, field: &mut IntentionField) {
        let aggregated = self.aggregate_weighted_average();
        field.update(aggregated, self.contributions.len());
    }

    /// Measure divergence among contributions.
    pub fn divergence(&self) -> f64 {
        if self.contributions.len() <= 1 {
            return 0.0;
        }
        let mean = self.aggregate_weighted_average();
        let total: f64 = self.contributions.iter()
            .map(|c| c.intention.distance_to(&mean))
            .sum();
        total / self.contributions.len() as f64
    }

    /// Number of contributing agents.
    pub fn len(&self) -> usize {
        self.contributions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contributions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_contribution() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        assert_eq!(agg.len(), 1);
    }

    #[test]
    fn test_replace_contribution() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![0.0, 1.0]), 2.0));
        assert_eq!(agg.len(), 1);
        assert!((agg.contributions[0].weight - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_average() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![2.0, 0.0]), 1.0));
        agg.add(AgentContribution::new(1, IntentionVector::new(vec![0.0, 2.0]), 1.0));
        let avg = agg.aggregate_weighted_average();
        assert!((avg.dims[0] - 1.0).abs() < 1e-10);
        assert!((avg.dims[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_divergence_zero() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        assert!((agg.divergence()).abs() < 1e-10);
    }

    #[test]
    fn test_remove_contribution() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 1.0));
        agg.remove(0);
        assert!(agg.is_empty());
    }

    #[test]
    fn test_apply_to_field() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![3.0, 4.0]), 1.0));
        let mut field = IntentionField::new(2);
        agg.apply_to_field(&mut field);
        assert_eq!(field.n_agents, 1);
        assert!((field.strength - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_aggregation() {
        let mut agg = ContributionAggregator::new();
        agg.add(AgentContribution::new(0, IntentionVector::new(vec![1.0, 0.0]), 0.5));
        agg.add(AgentContribution::new(1, IntentionVector::new(vec![0.0, 1.0]), 1.0));
        let consensus = agg.aggregate_consensus();
        // Should pick highest weight
        assert!((consensus.dims[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_contribution() {
        let c = AgentContribution::new(0, IntentionVector::new(vec![2.0, 3.0]), 0.5);
        let w = c.weighted();
        assert!((w.dims[0] - 1.0).abs() < 1e-10);
        assert!((w.dims[1] - 1.5).abs() < 1e-10);
    }
}
