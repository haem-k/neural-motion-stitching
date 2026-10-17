import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .resnet import Resnet1D
import nms.utils.utils as utils
import numpy as np
from nms.training.process import export_pos_feature_vis_np

class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, contras_weight, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.contras_weight = contras_weight
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.num_embed, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # codebook_size
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        activate = torch.sum(code_count > 0).float() / self.num_embed
        return perplexity, activate 
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z = rearrange(z, 'b c l -> b l c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1) 
        # look up the closest point for the indices
        encoding_indices = indices[:,-1] # [b*h*w]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = rearrange(z_q, 'b l c -> b c l').contiguous()
        # count
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        # return z_q, loss, (perplexity, min_encodings, encoding_indices)
        return z_q, encoding_indices, loss, perplexity # Match the ones that I have

    def forward_single(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z_flattened = z

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1) 
        # look up the closest point for the indices
        encoding_indices = indices[:,-1] # [b*h*w]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        # loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        loss_vq = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # z_q = rearrange(z_q, 'b l c -> b c l').contiguous()
        # count
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        loss_contras = torch.zeros([1]).to(z.device)
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                loss_contras = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss_contras = self.contras_weight * loss_contras

        # return z_q, loss, (perplexity, min_encodings, encoding_indices)
        return z_q, encoding_indices, loss_vq, loss_contras, perplexity # Match the ones that I have

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features

    
class CVQVAE(nn.Module):
    def __init__(self, config, statistics=None):
        super(CVQVAE, self).__init__()
        
        self.input_dim = config.input_dim
        self.input_len = config.input_len
        self.output_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        
        # buffer for stats
        self.register_buffer('mean', torch.zeros(self.input_dim))
        self.register_buffer('std', torch.zeros(self.input_dim))
        if statistics:
            self.mean = statistics["mean"]
            self.std = statistics["std"]
                
        # ! warning: reduced length hard-coded
        self.reduced_len = self.input_len // 4
        
        self.num_resnet = config.num_resnet
        self._encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            Resnet1D(self.hidden_dim // 2, self.num_resnet),

            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            Resnet1D(self.hidden_dim, self.num_resnet),

            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            Resnet1D(self.hidden_dim, self.num_resnet),
        )
        
        self.linear_dim = (self.input_len // 4) * self.hidden_dim
        self._encoder_linear = nn.Sequential(
            nn.Linear(self.linear_dim, self.linear_dim // 2),
            nn.GELU(),
            nn.Linear(self.linear_dim // 2, self.codebook_dim),
            nn.GELU()
        )
        
        self.quantizer = VectorQuantiser(self.codebook_size, self.codebook_dim, config.beta, config.w_contras, distance=config.distance, 
                                       anchor=config.anchor, first_batch=False, contras_loss=config.contras_loss)
        
        self._decoder_linear = nn.Sequential(
            nn.Linear(self.codebook_dim, self.linear_dim // 2),
            nn.GELU(),
            nn.Linear(self.linear_dim // 2, self.linear_dim),
        )
        self._decoder = nn.Sequential(
            Resnet1D(self.hidden_dim, self.num_resnet),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),

            Resnet1D(self.hidden_dim, self.num_resnet),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            
            Resnet1D(self.hidden_dim // 2, self.num_resnet),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(self.hidden_dim // 2, self.input_dim, kernel_size=3, stride=1, padding=1),
        )
    
    def encode(self, x):
        x = (x - self.mean) / self.std
        x = x.permute(0, 2, 1).float()
        x_enc = self._encoder(x)
        x_enc = x_enc.permute(0, 2, 1).float()
        
        batch_size = x_enc.shape[0]
        x_enc = torch.reshape(x_enc, [batch_size, -1])
        x_enc = self._encoder_linear(x_enc) # x: [N, L, C]

        x_q, encoding_indices, loss_vq, loss_contras, perplexity = self.quantizer.forward_single(x_enc)
        return x_enc, x_q, encoding_indices

    def decode(self, x_q):
        batch_size = x_q.shape[0]
        
        x_dec = self._decoder_linear(x_q)
        
        x_dec = torch.reshape(x_dec, [batch_size, self.reduced_len, -1])
        x_dec = x_dec.permute(0, 2, 1).float()
        x_dec = self._decoder(x_dec)
        
        x_out = x_dec.permute(0, 2, 1).float()
        x_out = x_out * self.std + self.mean
        return x_out

    def forward(self, x):
        x = (x - self.mean) / self.std
        
        # x: [N, L, C]
        x = x.permute(0, 2, 1).float()
        x_enc = self._encoder(x) # [N, C, L]
        x_enc = x_enc.permute(0, 2, 1).float() # [N, L, C]
        
        # flatten for linear
        reduced_len = x_enc.shape[1]
        batch_size = x_enc.shape[0]
        x_enc = torch.reshape(x_enc, [batch_size, -1])
        
        x_enc = self._encoder_linear(x_enc) # x: [N, codebook_dim]
        x_q, encoding_indices, loss_vq, loss_contras, perplexity = self.quantizer.forward_single(x_enc)        
        x_dec = self._decoder_linear(x_q)
        
        x_dec = torch.reshape(x_dec, [batch_size, reduced_len, -1])
        x_dec = x_dec.permute(0, 2, 1).float()
        x_dec = self._decoder(x_dec)
        
        x_out = x_dec.permute(0, 2, 1).float()
        x_out = x_out * self.std + self.mean
        return x_out, encoding_indices, loss_vq, loss_contras, perplexity

    def infer_codebook(self):
        print(f"\nDecoding all codes in codebook...")
        
        # decode all cluster codes
        with torch.no_grad():
            codebook_decode = self.quantizer.embedding.weight.clone()
            decoded = self.decode(codebook_decode)
        return decoded 
    
    def get_codebook(self):
        return self.quantizer.embedding.weight.clone().detach()
    
    def save_codebook(self, result_dir: str, datarep: dict):
        '''
        Saves codebok in npy format
        '''
        codebook = self.get_codebook().cpu().numpy()
        decoded = self.infer_codebook().cpu().numpy()
        
        codebook_dir = f"{result_dir}/codebook"
        utils.check_create_dir(codebook_dir)
        np.save(f"{codebook_dir}/codebook.npy", codebook)
        np.save(f"{codebook_dir}/codebook_decoded.npy", decoded) # save entire decoded result
        print("Saved codebook!")
        
        # export file for visualization
        export_pos_feature_vis_np(decoded, datarep, codebook_dir, "vis_codebook_decoded", flatten=False) 
