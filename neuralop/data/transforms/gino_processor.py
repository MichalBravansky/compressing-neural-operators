from neuralop.data.transforms.data_processors import DataProcessor
import torch

class GINOCFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an GINO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device='cuda'):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        # Turn a data dictionary returned by MeshDataModule's DictDataset
        # into the form expected by the GINO
        
        # input geometry: just vertices
        in_p = sample['vertices'].squeeze(0).to(self.device)
        latent_queries = sample['query_points'].squeeze(0).to(self.device)
        out_p = sample['vertices'].squeeze(0).to(self.device)
        f = sample['distance'].to(self.device)

        #Output data
        truth = sample['press'].squeeze(0).unsqueeze(-1)

        # Take the first 3586 vertices of the output mesh to correspond to pressure
        # if there are less than 3586 vertices, take the maximum number of truth points
        output_vertices = truth.shape[1]
        if out_p.shape[0] > output_vertices:
            out_p = out_p[:output_vertices,:]

        truth = truth.to(self.device)

        batch_dict = dict(input_geom=in_p,
                          latent_queries=latent_queries,
                          output_queries=out_p,
                          latent_features=f,
                          y=truth,
                          x=None)

        sample.update(batch_dict)
        return sample
    
    def postprocess(self, out, sample):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample['y'].squeeze(0))
            sample['y'] = y
        return out, sample
    
    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self
    
    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample 