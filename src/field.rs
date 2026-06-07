//! Core intention field struct.

/// A point in the intention space (e.g., a musical direction).
#[derive(Debug, Clone, PartialEq)]
pub struct IntentionVector {
    /// Dimension values (e.g., [tempo, harmony, dynamics, timbre]).
    pub dims: Vec<f64>,
}

impl IntentionVector {
    pub fn new(dims: Vec<f64>) -> Self {
        Self { dims }
    }

    pub fn zero(n: usize) -> Self {
        Self { dims: vec![0.0; n] }
    }

    pub fn dimension(&self) -> usize {
        self.dims.len()
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.dims.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Euclidean distance to another vector.
    pub fn distance_to(&self, other: &IntentionVector) -> f64 {
        self.dims.iter().zip(other.dims.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Add another vector.
    pub fn add(&self, other: &IntentionVector) -> IntentionVector {
        IntentionVector::new(
            self.dims.iter().zip(other.dims.iter()).map(|(a, b)| a + b).collect()
        )
    }

    /// Scale by a factor.
    pub fn scale(&self, factor: f64) -> IntentionVector {
        IntentionVector::new(self.dims.iter().map(|x| x * factor).collect())
    }

    /// Dot product.
    pub fn dot(&self, other: &IntentionVector) -> f64 {
        self.dims.iter().zip(other.dims.iter()).map(|(a, b)| a * b).sum()
    }

    /// Normalized version.
    pub fn normalized(&self) -> IntentionVector {
        let n = self.norm();
        if n < 1e-10 {
            return self.clone();
        }
        self.scale(1.0 / n)
    }
}

/// The shared intention field.
#[derive(Debug, Clone)]
pub struct IntentionField {
    /// Dimensionality of the intention space.
    pub dimensions: usize,
    /// Current field state.
    pub state: IntentionVector,
    /// Number of contributing agents.
    pub n_agents: usize,
    /// Field strength (magnitude of current intention).
    pub strength: f64,
}

impl IntentionField {
    /// Create an empty intention field.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            state: IntentionVector::zero(dimensions),
            n_agents: 0,
            strength: 0.0,
        }
    }

    /// Update the field state.
    pub fn update(&mut self, new_state: IntentionVector, n_agents: usize) {
        self.strength = new_state.norm();
        self.state = new_state;
        self.n_agents = n_agents;
    }

    /// Clear the field.
    pub fn clear(&mut self) {
        self.state = IntentionVector::zero(self.dimensions);
        self.n_agents = 0;
        self.strength = 0.0;
    }

    /// Check if the field is empty (no agents contributing).
    pub fn is_empty(&self) -> bool {
        self.n_agents == 0
    }

    /// Distance from a given intention to the current field state.
    pub fn distance_from(&self, intention: &IntentionVector) -> f64 {
        self.state.distance_to(intention)
    }

    /// Alignment of a given intention with the field (cosine similarity).
    pub fn alignment(&self, intention: &IntentionVector) -> f64 {
        let norm_product = self.state.norm() * intention.norm();
        if norm_product < 1e-10 {
            return 0.0;
        }
        self.state.dot(intention) / norm_product
    }

    /// Merge another field into this one.
    pub fn merge(&mut self, other: &IntentionField) {
        let total_agents = self.n_agents + other.n_agents;
        if total_agents == 0 {
            return;
        }
        let w1 = self.n_agents as f64 / total_agents as f64;
        let w2 = other.n_agents as f64 / total_agents as f64;
        let merged = self.state.scale(w1).add(&other.state.scale(w2));
        self.update(merged, total_agents);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_field() {
        let f = IntentionField::new(3);
        assert_eq!(f.dimensions, 3);
        assert!(f.is_empty());
    }

    #[test]
    fn test_vector_distance() {
        let a = IntentionVector::new(vec![1.0, 0.0]);
        let b = IntentionVector::new(vec![0.0, 1.0]);
        assert!((a.distance_to(&b) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_alignment() {
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![1.0, 0.0]), 1);
        let aligned = IntentionVector::new(vec![1.0, 0.0]);
        assert!((field.alignment(&aligned) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge_fields() {
        let mut f1 = IntentionField::new(2);
        f1.update(IntentionVector::new(vec![2.0, 0.0]), 2);
        let mut f2 = IntentionField::new(2);
        f2.update(IntentionVector::new(vec![0.0, 4.0]), 2);
        f1.merge(&f2);
        assert_eq!(f1.n_agents, 4);
    }

    #[test]
    fn test_normalized() {
        let v = IntentionVector::new(vec![3.0, 4.0]);
        let n = v.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clear() {
        let mut f = IntentionField::new(2);
        f.update(IntentionVector::new(vec![1.0, 1.0]), 3);
        f.clear();
        assert!(f.is_empty());
        assert!((f.strength).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = IntentionVector::new(vec![1.0, 2.0]);
        let b = IntentionVector::new(vec![3.0, 4.0]);
        assert!((a.dot(&b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_anti_alignment() {
        let mut field = IntentionField::new(2);
        field.update(IntentionVector::new(vec![1.0, 0.0]), 1);
        let anti = IntentionVector::new(vec![-1.0, 0.0]);
        assert!((field.alignment(&anti) - (-1.0)).abs() < 1e-10);
    }
}
