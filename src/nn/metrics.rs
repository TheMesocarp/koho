#[derive(Debug)]
/// Metrics collected each epoch
pub struct EpochMetrics {
    pub epoch: usize,
    pub loss: f32,
    // will probs need to add some extra metrics beyond loss
}

impl EpochMetrics {
    pub fn new(epoch: usize, loss: f32) -> Self {
        Self { epoch, loss }
    }
}

#[derive(Debug)]
/// Metrics collected during a `train()` run
pub struct TrainingMetrics {
    pub epochs: Vec<EpochMetrics>,
    pub final_loss: f32,
    pub total_epochs: usize,
}

impl TrainingMetrics {
    pub fn new(total_epochs: usize) -> Self {
        Self {
            epochs: Vec::new(),
            final_loss: f32::MAX,
            total_epochs,
        }
    }

    pub fn push(&mut self, epoch_metrics: EpochMetrics) {
        self.final_loss = epoch_metrics.loss;
        self.epochs.push(epoch_metrics);
    }

    pub fn to_loss_list(&self) -> Vec<(usize, f32)> {
        self.epochs
            .iter()
            .map(|x| (x.epoch, x.loss))
            .collect::<Vec<_>>()
    }
}
