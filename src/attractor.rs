//! Musical attractors in the intention field.

use crate::field::{IntentionField, IntentionVector};

/// A musical attractor: a preferred direction in intention space.
#[derive(Debug, Clone)]
pub struct Attractor {
    pub id: usize,
    pub position: IntentionVector,
    /// Strength of attraction (pull).
    pub strength: f64,
    /// Radius of influence.
    pub radius: f64,
}

impl Attractor {
    pub fn new(id: usize, position: IntentionVector, strength: f64, radius: f64) -> Self {
        Self { id, position, strength, radius }
    }

    /// Compute the pull force on a given intention vector.
    pub fn pull(&self, intention: &IntentionVector) -> IntentionVector {
        let dist = self.position.distance_to(intention);
        if dist > self.radius || dist < 1e-10 {
            return IntentionVector::zero(intention.dimension());
        }
        // Force proportional to strength and inversely to distance
        let force_magnitude = self.strength * (1.0 - dist / self.radius);
        let direction = self.position.add(&intention.scale(-1.0)).normalized();
        direction.scale(force_magnitude)
    }

    /// Whether an intention is within the attractor's influence radius.
    pub fn influences(&self, intention: &IntentionVector) -> bool {
        self.position.distance_to(intention) <= self.radius
    }
}

/// Collection of attractors.
#[derive(Debug, Clone)]
pub struct AttractorField {
    pub attractors: Vec<Attractor>,
}

impl Default for AttractorField {
    fn default() -> Self {
        Self::new()
    }
}

impl AttractorField {
    pub fn new() -> Self {
        Self { attractors: Vec::new() }
    }

    /// Add an attractor.
    pub fn add(&mut self, attractor: Attractor) {
        self.attractors.push(attractor);
    }

    /// Compute the net force from all attractors on the field.
    pub fn net_force(&self, field: &IntentionField) -> IntentionVector {
        if self.attractors.is_empty() {
            return IntentionVector::zero(field.dimensions);
        }
        let mut total = IntentionVector::zero(field.dimensions);
        for a in &self.attractors {
            total = total.add(&a.pull(&field.state));
        }
        total
    }

    /// Find the strongest attractor influencing the field.
    pub fn dominant_attractor(&self, field: &IntentionField) -> Option<&Attractor> {
        self.attractors.iter()
            .filter(|a| a.influences(&field.state))
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of attractors.
    pub fn len(&self) -> usize {
        self.attractors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.attractors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attractor_pull() {
        let a = Attractor::new(0, IntentionVector::new(vec![10.0, 0.0]), 1.0, 20.0);
        let v = IntentionVector::new(vec![5.0, 0.0]);
        let pull = a.pull(&v);
        assert!(pull.dims[0] > 0.0, "pull should be toward attractor");
    }

    #[test]
    fn test_attractor_outside_radius() {
        let a = Attractor::new(0, IntentionVector::new(vec![10.0, 0.0]), 1.0, 2.0);
        let v = IntentionVector::new(vec![0.0, 0.0]);
        let pull = a.pull(&v);
        assert!((pull.norm()).abs() < 1e-10, "no pull outside radius");
    }

    #[test]
    fn test_influences() {
        let a = Attractor::new(0, IntentionVector::new(vec![5.0, 0.0]), 1.0, 3.0);
        assert!(a.influences(&IntentionVector::new(vec![7.0, 0.0])));
        assert!(!a.influences(&IntentionVector::new(vec![0.0, 0.0])));
    }

    #[test]
    fn test_net_force() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![10.0, 0.0]), 1.0, 20.0));
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![5.0, 0.0]), 1);
        let force = af.net_force(&field);
        assert!(force.norm() > 0.0);
    }

    #[test]
    fn test_dominant_attractor() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![5.0, 0.0]), 1.0, 10.0));
        af.add(Attractor::new(1, IntentionVector::new(vec![5.0, 0.0]), 2.0, 10.0));
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![5.0, 0.0]), 1);
        let dominant = af.dominant_attractor(&field);
        assert!(dominant.is_some());
        assert_eq!(dominant.unwrap().id, 1);
    }

    #[test]
    fn test_attractor_zero_distance_pull() {
        let a = Attractor::new(0, IntentionVector::new(vec![5.0, 5.0]), 2.0, 10.0);
        let at_center = IntentionVector::new(vec![5.0, 5.0]);
        let pull = a.pull(&at_center);
        assert!(pull.norm() < 1e-10, "no pull when at attractor center");
    }

    #[test]
    fn test_attractor_len() {
        let mut af = AttractorField::new();
        assert!(af.is_empty());
        af.add(Attractor::new(0, IntentionVector::new(vec![0.0, 0.0]), 1.0, 1.0));
        af.add(Attractor::new(1, IntentionVector::new(vec![1.0, 1.0]), 1.0, 1.0));
        assert_eq!(af.len(), 2);
        assert!(!af.is_empty());
    }

    #[test]
    fn test_net_force_empty() {
        let af = AttractorField::new();
        let field = IntentionField::new(2);
        let force = af.net_force(&field);
        assert!(force.norm() < 1e-10);
    }
}
