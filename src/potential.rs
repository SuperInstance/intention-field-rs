//! Potential energy landscape for the intention field.

use crate::field::{IntentionField, IntentionVector};
use crate::attractor::AttractorField;

/// A point in the potential landscape.
#[derive(Debug, Clone)]
pub struct LandscapePoint {
    pub position: IntentionVector,
    pub potential: f64,
}

impl LandscapePoint {
    pub fn new(position: IntentionVector, potential: f64) -> Self {
        Self { position, potential }
    }
}

/// Potential energy landscape.
#[derive(Debug, Clone)]
pub struct PotentialLandscape {
    /// Attractors create wells in the landscape.
    pub attractor_field: AttractorField,
    /// Repulsion between agents (disagreement penalty).
    pub disagreement_penalty: f64,
}

impl PotentialLandscape {
    pub fn new(attractor_field: AttractorField, disagreement_penalty: f64) -> Self {
        Self { attractor_field, disagreement_penalty }
    }

    /// Compute the potential energy at a given intention.
    pub fn potential_at(&self, intention: &IntentionVector, field: &IntentionField) -> f64 {
        let mut potential = 0.0;

        // Attractors create potential wells (negative energy)
        for a in &self.attractor_field.attractors {
            let dist = a.position.distance_to(intention);
            if dist < a.radius {
                // Quadratic well
                let depth = a.strength * (1.0 - (dist / a.radius).powi(2));
                potential -= depth;
            }
        }

        // Distance from current field state adds penalty
        let field_distance = field.distance_from(intention);
        potential += self.disagreement_penalty * field_distance;

        potential
    }

    /// Compute the gradient (steepest descent direction) at a point.
    pub fn gradient(&self, intention: &IntentionVector, field: &IntentionField) -> IntentionVector {
        let eps = 1e-6;
        let dims = intention.dimension();
        let mut grad_dims = vec![0.0; dims];
        let base_potential = self.potential_at(intention, field);

        for i in 0..dims {
            let mut shifted = intention.dims.clone();
            shifted[i] += eps;
            let shifted_vec = IntentionVector::new(shifted);
            let shifted_potential = self.potential_at(&shifted_vec, field);
            grad_dims[i] = (shifted_potential - base_potential) / eps;
        }

        IntentionVector::new(grad_dims)
    }

    /// Find the nearest local minimum.
    pub fn nearest_minimum(
        &self,
        start: &IntentionVector,
        field: &IntentionField,
        learning_rate: f64,
        max_steps: usize,
    ) -> IntentionVector {
        let mut current = start.clone();
        for _ in 0..max_steps {
            let grad = self.gradient(&current, field);
            let new_pos = current.add(&grad.scale(-learning_rate));
            if new_pos.distance_to(&current) < 1e-8 {
                break;
            }
            current = new_pos;
        }
        current
    }

    /// Sample the landscape on a grid (1D projection for 2+ dimensions).
    pub fn sample_1d(
        &self,
        field: &IntentionField,
        dim: usize,
        range: (f64, f64),
        n_points: usize,
    ) -> Vec<LandscapePoint> {
        let step = (range.1 - range.0) / (n_points - 1).max(1) as f64;
        (0..n_points).map(|i| {
            let mut dims = field.state.dims.clone();
            if dim < dims.len() {
                dims[dim] = range.0 + i as f64 * step;
            }
            let pos = IntentionVector::new(dims);
            let pot = self.potential_at(&pos, field);
            LandscapePoint::new(pos, pot)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attractor::Attractor;

    fn make_field() -> IntentionField {
        let mut f = IntentionField::new(2);
        f.update(IntentionVector::new(vec![5.0, 5.0]), 3);
        f
    }

    #[test]
    fn test_potential_at_attractor() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![5.0, 5.0]), 2.0, 10.0));
        let landscape = PotentialLandscape::new(af, 0.1);
        let field = make_field();
        let pot = landscape.potential_at(&IntentionVector::new(vec![5.0, 5.0]), &field);
        // At the attractor center, potential should be negative
        assert!(pot < 0.0, "potential at attractor center should be negative, got {}", pot);
    }

    #[test]
    fn test_gradient_descent() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![10.0, 10.0]), 3.0, 20.0));
        let landscape = PotentialLandscape::new(af, 0.1);
        let field = make_field();
        let start = IntentionVector::new(vec![5.0, 5.0]);
        let minimum = landscape.nearest_minimum(&start, &field, 0.1, 100);
        // Should move toward attractor
        assert!(minimum.dims[0] > start.dims[0]);
    }

    #[test]
    fn test_sample_1d() {
        let af = AttractorField::new();
        let landscape = PotentialLandscape::new(af, 0.5);
        let field = make_field();
        let samples = landscape.sample_1d(&field, 0, (0.0, 10.0), 11);
        assert_eq!(samples.len(), 11);
    }

    #[test]
    fn test_landscape_no_attractors() {
        let af = AttractorField::new();
        let landscape = PotentialLandscape::new(af, 0.0);
        let field = make_field();
        let pot = landscape.potential_at(&IntentionVector::new(vec![5.0, 5.0]), &field);
        assert!((pot).abs() < 1e-10);
    }

    #[test]
    fn test_disagreement_penalty() {
        let af = AttractorField::new();
        let landscape = PotentialLandscape::new(af, 2.0);
        let field = make_field();
        let near = landscape.potential_at(&IntentionVector::new(vec![5.0, 5.0]), &field);
        let far = landscape.potential_at(&IntentionVector::new(vec![50.0, 50.0]), &field);
        assert!(far > near, "farther point should have higher penalty");
    }

    #[test]
    fn test_gradient_magnitude() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![10.0, 10.0]), 5.0, 20.0));
        let landscape = PotentialLandscape::new(af, 0.1);
        let field = make_field();
        let grad = landscape.gradient(&IntentionVector::new(vec![0.0, 0.0]), &field);
        assert!(grad.norm() > 0.0, "gradient should be nonzero near attractor edge");
    }

    #[test]
    fn test_sample_1d_varying_potential() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![5.0, 5.0]), 10.0, 3.0));
        let landscape = PotentialLandscape::new(af, 0.01);
        let field = make_field();
        let samples = landscape.sample_1d(&field, 0, (0.0, 10.0), 21);
        let min_pot = samples.iter().map(|s| s.potential).fold(f64::INFINITY, f64::min);
        assert!(min_pot < 0.0, "should find negative potential near attractor");
    }

    #[test]
    fn test_nearest_minimum_converges() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![5.0, 5.0]), 5.0, 10.0));
        let landscape = PotentialLandscape::new(af, 0.01);
        let field = make_field();
        let start = IntentionVector::new(vec![4.0, 4.0]);
        let min = landscape.nearest_minimum(&start, &field, 0.1, 200);
        let dist = min.distance_to(&IntentionVector::new(vec![5.0, 5.0]));
        assert!(dist < 1.0, "should converge near attractor, dist={}", dist);
    }

    #[test]
    fn test_potential_inside_vs_outside_radius() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![5.0, 5.0]), 10.0, 3.0));
        let landscape = PotentialLandscape::new(af, 0.0);
        let field = make_field();
        let inside = landscape.potential_at(&IntentionVector::new(vec![5.0, 5.0]), &field);
        let outside = landscape.potential_at(&IntentionVector::new(vec![10.0, 10.0]), &field);
        assert!(inside < outside, "inside attractor should have lower potential");
    }

    #[test]
    fn test_multiple_attractors_additive() {
        let mut af = AttractorField::new();
        af.add(Attractor::new(0, IntentionVector::new(vec![3.0, 3.0]), 5.0, 5.0));
        af.add(Attractor::new(1, IntentionVector::new(vec![7.0, 7.0]), 5.0, 5.0));
        let landscape = PotentialLandscape::new(af, 0.0);
        let field = make_field();
        let midpoint = landscape.potential_at(&IntentionVector::new(vec![5.0, 5.0]), &field);
        let single = {
            let mut af2 = AttractorField::new();
            af2.add(Attractor::new(0, IntentionVector::new(vec![3.0, 3.0]), 5.0, 5.0));
            let ls = PotentialLandscape::new(af2, 0.0);
            ls.potential_at(&IntentionVector::new(vec![5.0, 5.0]), &field)
        };
        assert!(midpoint < single, "two attractors should create deeper well");
    }
}
