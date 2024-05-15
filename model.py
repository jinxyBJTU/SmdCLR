import torch
from torch import nn
import torch.nn.functional as F


class base_Model(nn.Module):
    def __init__(self, args, configs, fine_tune=False):
        # simclr 个体判别，logits的维度是128，需要多加一个mlp
        # cpc\tstcc 对比判别，最终是用rnn之前的encoding作为样本的表示向量的，即logits的linear预训练完全不使用，fine-tune再使用
        super(base_Model, self).__init__()
        self.if_gcn = False

        if 'GCN' in args.model_name:
            self.if_gcn = True

        self.fine_tune = fine_tune

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8,
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.logits = nn.Linear( configs.features_len * configs.final_out_channels, configs.final_out_channels )

        self.projection1 = nn.Linear( configs.final_out_channels, configs.final_out_channels )
        self.projection2 = nn.Linear( configs.final_out_channels, configs.final_out_channels )
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.classifier = nn.Linear( configs.final_out_channels, configs.num_classes )

    def forward(self, x1, x2=None):
        
        if self.fine_tune:
            x = self.conv_block1(x1)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x_flat = x.reshape(x.shape[0], -1)
            mean_embeddings = self.logits(x_flat)
            logits = self.classifier(mean_embeddings)
            return logits, mean_embeddings
        
        else:
            
            embeddings = []
            for x in [x1, x2]:
                x = self.conv_block1(x)
                x = self.conv_block2(x)
                x = self.conv_block3(x)
                
                x_flat = x.reshape(x.shape[0], -1)
                mean_embeddings = self.logits(x_flat)
                embeddings.append(mean_embeddings)
            
            embedding1, embedding2 = embeddings[0],embeddings[1]
            encoder_embeddings = torch.cat([embedding1, embedding2], dim=0) # 2B,F

            encoder_sim_matrix = self.cosine_similarity(encoder_embeddings.unsqueeze(1), encoder_embeddings.unsqueeze(0))
            encoder_sim_matrix = F.softmax(encoder_sim_matrix, dim=1)

            if self.if_gcn:
                projection_embeddings = self.projection1(torch.matmul(encoder_sim_matrix, encoder_embeddings))
                projection_embeddings = self.projection2(F.relu(torch.matmul(encoder_sim_matrix, projection_embeddings)))
                
            else:
                projection_embeddings = self.projection2(F.relu(self.projection1(encoder_embeddings)))
        
            encoder_classification_results =  self.classifier(encoder_embeddings)
            projection_classification_results =  self.classifier(projection_embeddings)
               
            return encoder_sim_matrix, encoder_embeddings, projection_embeddings, encoder_classification_results, projection_classification_results
