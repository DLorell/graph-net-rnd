import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.densenet121.features(x)
        x = F.relu(x)
        return x


class Attention(nn.Module):

    def __init__(self, k_size, v_size, affine_size=512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_size, bias=False)
        self.affine_v = nn.Linear(v_size, affine_size, bias=False)
        self.affine = nn.Linear(affine_size, 1, bias=False)

    def forward(self, k, v):
        # k: batch size x hidden size
        # v: batch size x spatial size x hidden size
        # z: batch size x spatial size
        # TODO other ways of attention?
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class SentSAT(nn.Module):

    def __init__(self, vocab_size, feat_size=1024, embed_size=256, hidden_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.atten = Attention(hidden_size, feat_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.init_sent_h = nn.Linear(2 * feat_size, hidden_size)
        self.init_sent_c = nn.Linear(2 * feat_size, hidden_size)
        self.sent_lstm = nn.LSTMCell(2 * feat_size, hidden_size)
        self.word_lstm = nn.LSTMCell(embed_size + hidden_size + 2 * feat_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, cnn_feats1, cnn_feats2, captions=None, update_masks=None, stop_id=None, max_sents=10, max_len=30):
        batch_size = cnn_feats1.size(0)
        if captions is not None:
            num_sents = captions.size(1)
            seq_len = captions.size(2)
        else:
            num_sents = max_sents
            seq_len = max_len

        cnn_feats1_t = cnn_feats1.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        cnn_feats2_t = cnn_feats2.view(batch_size, self.feat_size, -1).permute(0, 2, 1)
        global_feats1 = cnn_feats1.mean(dim=(2, 3))
        global_feats2 = cnn_feats2.mean(dim=(2, 3))
        sent_h = self.init_sent_h(torch.cat((global_feats1, global_feats2), dim=1))
        sent_c = self.init_sent_c(torch.cat((global_feats1, global_feats2), dim=1))
        word_h = cnn_feats1.new_zeros((batch_size, self.hidden_size), dtype=torch.float)
        word_c = cnn_feats1.new_zeros((batch_size, self.hidden_size), dtype=torch.float)

        logits = cnn_feats1.new_zeros((batch_size, num_sents, seq_len, self.vocab_size), dtype=torch.float)

        if captions is not None:
            embeddings = self.embed(captions)

            for k in range(num_sents):
                context1, alpha1 = self.atten(sent_h, cnn_feats1_t)
                context2, alpha2 = self.atten(sent_h, cnn_feats2_t)
                context = torch.cat((context1, context2), dim=1)
                sent_h, sent_c = self.sent_lstm(context, (sent_h, sent_c))
                seq_len_k = update_masks[:, k].sum(dim=1).max().item()

                for t in range(seq_len_k):
                    batch_mask = update_masks[:, k, t]
                    word_h_, word_c_ = self.word_lstm(
                        torch.cat((embeddings[batch_mask, k, t], sent_h[batch_mask], context[batch_mask]), dim=1),
                        (word_h[batch_mask], word_c[batch_mask]))
                    indices = [*batch_mask.unsqueeze(1).repeat(1, self.hidden_size).nonzero().t()]
                    word_h = word_h.index_put(indices, word_h_.view(-1))
                    word_c = word_c.index_put(indices, word_c_.view(-1))
                    logits[batch_mask, k, t] = self.fc(self.dropout(word_h[batch_mask]))

            return logits

        else:
            x_t = cnn_feats1.new_full((batch_size,), 1, dtype=torch.long)

            for k in range(num_sents):
                context1, alpha1 = self.atten(sent_h, cnn_feats1_t)
                context2, alpha2 = self.atten(sent_h, cnn_feats2_t)
                context = torch.cat((context1, context2), dim=1)
                sent_h, sent_c = self.sent_lstm(context, (sent_h, sent_c))

                for t in range(seq_len):
                    embedding = self.embed(x_t)
                    word_h, word_c = self.word_lstm(torch.cat((embedding, sent_h, context), dim=1), (word_h, word_c))
                    logit = self.fc(word_h)
                    x_t = logit.argmax(dim=1)
                    logits[:, k, t] = logit

                    if x_t[0] == stop_id:
                        break

            return logits.argmax(dim=3)


class Encoder2Decoder(nn.Module):

    def __init__(self, num_classes, vocab_size, feat_size=1024, embed_size=256, hidden_size=512):
        super().__init__()
        self.encoder = Encoder(num_classes)
        self.decoder = SentSAT(vocab_size, feat_size, embed_size, hidden_size)

    def forward(self, images1, images2, captions=None, update_masks=None, stop_id=None, max_sents=10, max_len=30):
        cnn_feats1 = self.encoder(images1)
        cnn_feats2 = self.encoder(images2)
        return self.decoder(cnn_feats1, cnn_feats2, captions, update_masks, stop_id, max_sents, max_len)