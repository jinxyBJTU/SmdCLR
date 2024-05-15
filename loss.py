import torch
import numpy as np
import torch.nn.functional as F
import copy
from sklearn.cluster import KMeans
# 检查协方差矩阵是否是对称半正定
def check_covariance_matrix(cov):
    # 检查对称性
    if not torch.allclose(cov, cov.T):
        return False
    # 检查正定性
    eigvals = torch.linalg.eigvals(cov)
    
    if not (eigvals.real >= 0).all():
        return False
    return True

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none") # none sum

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def hand_cross_entropy_loss(self, input, isinstance_discri_labels, sim_matrix=None):
        # Compute log-softmax of input tensor along class dimension
        nll_loss = torch.nn.NLLLoss(reduction='none')

        softmax = F.softmax(input, dim=1)
        log_softmax = torch.log(softmax)

        # psedo_laels = torch.zeros_like(softmax).to(self.device).float()
        # psedo_laels[row_indices, indices] = values
        # # psedo_laels[:,0] = 1

        if sim_matrix!=None:
            # isinstance_discri_labels = isinstance_discri_labels.to(torch.int).float()
            # num_of_samples_in_multi = torch.sum(isinstance_discri_labels, dim=-1)
            
            ce_2batch_all = torch.sum( (-log_softmax)*(sim_matrix),  dim=-1 )
        else:
            ce_2batch_all = nll_loss( log_softmax, isinstance_discri_labels) # B

        # ce_2batch_all = confidence*(torch.sum( softmax_copy*(-log_softmax),  dim=-1 )/num_of_samples_in_multi )  +  (1-confidence)*nll_loss( log_softmax, target)
        # ce_2batch_all = (1-confidence)*nll_loss( log_softmax, target)
        # 需要时可以统计一下内容：
        # num_of_same = torch.sum(same_class_mask.type(torch.int),dim=-1)
        # num_of_diff = torch.sum(1-same_class_mask.type(torch.int),dim=-1)
        # if sim_matrix!=None:
        #     ce_2batch_same = torch.sum(((-log_softmax)*(sim_matrix)).masked_fill(same_class_mask == False, 0), dim=-1 ) / num_of_same
        #     ce_2batch_diff = torch.sum(((-log_softmax)*(sim_matrix)).masked_fill(same_class_mask == True, 0), dim=-1 ) / num_of_diff
        # else:
        #     ce_2batch_same = torch.sum((-log_softmax).masked_fill(same_class_mask == False, 0), dim=-1 ) / num_of_same
        #     ce_2batch_diff = torch.sum((-log_softmax).masked_fill(same_class_mask == True, 0), dim=-1 ) / num_of_diff

        # if sim_matrix!=None:
        #     sim_loss = torch.sum( (sim_matrix)*(same_class_mask.to(torch.int)),  dim=-1 )/ num_of_same
        #     return sim_loss, ce_2batch_all, ce_2batch_same, ce_2batch_diff
        # else:
        return ce_2batch_all, ce_2batch_all, ce_2batch_all
    
    def forward(self, representations, sim_matrix=None):

        if sim_matrix!=None:
            sim_matrix = sim_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
            ones = torch.tensor([1]).to(self.device).reshape(1,1).repeat(2 * self.batch_size, 1)
            sim_matrix = torch.cat((ones, sim_matrix), dim=1)  
            
        # representations = torch.cat([zjs, zis], dim=0) # 2B,F
        similarity_matrix = self.similarity_function(representations, representations)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) # (2B,1)

        # same_class_mask = same_class_mask.to(self.device)
        # same_class_mask = same_class_mask[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)  # 2n-2
        # trues = torch.tensor([True]).to(self.device).reshape(1,1).repeat(2 * self.batch_size, 1)
        # same_class_mask = torch.cat((trues, same_class_mask), dim=1)                
       
        negatives_all = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1) # (2B, 2B-2) 
        logits_all = torch.cat((positives, negatives_all), dim=1)                                         # (2B, 2B-1)
        logits_all /= self.temperature

        isinstance_discri_labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # isinstance_discri_labels = torch.tensor([False]).to(self.device).reshape(1,1).repeat(2 * self.batch_size, 2 * self.batch_size-2)
        # isinstance_discri_labels = torch.cat((trues, isinstance_discri_labels), dim=1)
        # labels = torch.ones(2 * self.batch_size).to(self.device).long()
        # labels = torch.cat((labels.unsqueeze(-1), torch.zeros((2 * self.batch_size, 2 * self.batch_size - 2)).to(self.device).long()), dim=1)      

        # if sim_matrix!=None:
        #     sim_loss, ce_2batch_all, ce_2batch_same, ce_2batch_diff = self.hand_cross_entropy_loss(logits_all, isinstance_discri_labels, sim_matrix, same_class_mask) # # loss = self.criterion(logits, labels)
        #     return  np.stack([sim_loss.detach().cpu()],axis=1), \
        #             torch.mean(ce_2batch_all),  \
        #             np.stack([  ce_2batch_all.detach().cpu(),
        #                         ce_2batch_same.detach().cpu(),
        #                         ce_2batch_diff.detach().cpu()],axis=1)
        # else:
        ce_2batch_all, ce_2batch_same, ce_2batch_diff = self.hand_cross_entropy_loss(logits_all, isinstance_discri_labels, sim_matrix) # # loss = self.criterion(logits, labels)
        return torch.mean(ce_2batch_all),  np.stack([ce_2batch_all.detach().cpu(),
                                        ce_2batch_same.detach().cpu(),
                                        ce_2batch_diff.detach().cpu()],axis=1)


class Class_disc(torch.nn.Module):

    def __init__(self, device, temperature):
        super(Class_disc, self).__init__()
        self.temperature = temperature
        self.device = device
        # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none") # none sum

    def _get_similarity_function(self):
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity
    
    # def _get_correlated_mask(self):
    #     diag = np.eye(2 * self.batch_size)
    #     l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
    #     l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
    #     mask = torch.from_numpy((diag + l1 + l2))
    #     mask = (1 - mask).type(torch.bool)
    #     return mask.to(self.device)

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    
    def hand_cross_entropy_loss(self, input, isinstance_discri_labels, sim_matrix=None,):

        nll_loss = torch.nn.NLLLoss(reduction='none')
        softmax = F.softmax(input, dim=1)
        log_softmax = torch.log(softmax)
        if sim_matrix!=None:
            ce_2batch_all = torch.sum( (-log_softmax)*(sim_matrix),  dim=-1 )
        else:
            ce_2batch_all = nll_loss( log_softmax, isinstance_discri_labels) # B

        return ce_2batch_all
    
   
    def forward(self, representations, classifier_weight, labeled_idx, labels):
        """
        representations: 2B, F
        classifier_weight: C, F
        """

        labeled_labels = labels[labeled_idx]
        labeled_representations = representations[labeled_idx]
        logits = self.similarity_function(labeled_representations, classifier_weight)
        loss = self.criterion(logits, labeled_labels)
        return torch.mean(loss)
        # print(representations.shape)
        # print(classifier_weight.shape)
        # print(labeled_idx)
        # print(len(labeled_idx))
        # print(labeled_labels)
        # exit()


        # d = representations.shape[-1]
        # majority_samples, minority_samples = self.get_labeled_samples(representations, labels)

        ### KL 散度计算
        # mean_maj = torch.mean(majority_samples, dim=0)
        # cov_maj = torch.diag(torch.var(majority_samples, dim=0))
        # mean_min = torch.mean(minority_samples, dim=0)
        # cov_min = torch.diag(torch.var(minority_samples, dim=0))
        # cov2_inv = torch.inverse(cov_min)
        
        # # kl_div = 0.5 * (torch.trace(torch.matmul(cov2_inv, cov_maj)) + torch.matmul(torch.matmul(mean_min - mean_maj, cov2_inv), mean_min - mean_maj) - d + torch.log(torch.det(cov_min) / torch.det(cov_maj)))
        # term1 = torch.trace(torch.matmul(torch.inverse(cov2_inv), cov_maj))
        # term2 = torch.matmul(torch.matmul(mean_min - mean_maj, cov2_inv), mean_min - mean_maj)
        # term3 = -d
        # sign, logdet = torch.slogdet(cov_min / cov_maj)
        # term4 = sign * logdet
        # kl_div = 0.5 * (term1 + term2 + term3 + term4)
        
       
#  def get_labeled_samples(self, encoder_embeddings, labels, ratio = 0.02):
        
#         batchsize = labels.shape[0]
#         oversampling_numbers = int(batchsize * ratio) # 12
        
#         all_idx = list(range(batchsize))

#         num_of_labels = torch.max(labels)+1
#         counts = [(labels == i).sum() for i in range(num_of_labels)]
#         max_idx = counts.index(np.max(counts))

#         maj_samples_idx = [idx for idx, x in enumerate(labels) if x == max_idx]
#         maj_samples_idx = maj_samples_idx[:oversampling_numbers]

#         min_samples_idx = []
#         for cur_minority_idx in range(num_of_labels):  
#             if cur_minority_idx==max_idx:
#                 continue
#             cur_samples_idx = [idx for idx, x in enumerate(labels) if x == cur_minority_idx]
#             num_of_cur_samples = len(cur_samples_idx)
#             if num_of_cur_samples == 0:
#                 continue
#             if num_of_cur_samples<oversampling_numbers:
#                 min_samples_idx.extend(cur_samples_idx)
#             else:
#                 min_samples_idx.extend(cur_samples_idx[:oversampling_numbers])


#         maj_samples_idx = maj_samples_idx + [ x+batchsize for x in maj_samples_idx ]
#         min_samples_idx = min_samples_idx + [ x+batchsize for x in min_samples_idx ]

#         majority_samples = encoder_embeddings[maj_samples_idx]
#         minority_samples = encoder_embeddings[min_samples_idx]

#         return majority_samples, minority_samples
    