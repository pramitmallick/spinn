import itertools

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal

from spinn.util.blocks import Embed, Linear, MLP
from spinn.util.blocks import bundle, lstm, to_gpu, unbundle
from spinn.util.blocks import LayerNormalization
from spinn.util.misc import Example, Vocab
from spinn.util.catalan import ShiftProbabilities
import sys
from spinn.spinn_core_model import build_model as spinn_builder
from spinn.plain_rnn import build_model as rnn_builder

def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args, target_vocabulary=None, **kwargs):
    
    return NMTModel(
        model_dim=FLAGS.model_dim,
        word_embedding_dim=FLAGS.word_embedding_dim,
        vocab_size=vocab_size,
        initial_embeddings=initial_embeddings,
        fine_tune_loaded_embeddings=FLAGS.fine_tune_loaded_embeddings,
        num_classes=num_classes,
        embedding_keep_rate=FLAGS.embedding_keep_rate,
        tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
        transition_weight=FLAGS.transition_weight,
        use_sentence_pair=False,
        lateral_tracking=FLAGS.lateral_tracking,
        tracking_ln=FLAGS.tracking_ln,
        use_tracking_in_composition=FLAGS.use_tracking_in_composition,
        predict_use_cell=FLAGS.predict_use_cell,
        use_difference_feature=FLAGS.use_difference_feature,
        use_product_feature=FLAGS.use_product_feature,
        classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
        mlp_dim=FLAGS.mlp_dim,
        num_mlp_layers=FLAGS.num_mlp_layers,
        mlp_ln=FLAGS.mlp_ln,
        context_args=context_args,
        composition_args=composition_args,
        with_attention=FLAGS.with_attention,
        data_type=FLAGS.data_type,
        target_vocabulary=target_vocabulary,
        onmt_module=FLAGS.onmt_file_path,
        flags=FLAGS,
        data_manager=data_manager
    )


class NMTModel(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 fine_tune_loaded_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 lateral_tracking=None,
                 tracking_ln=None,
                 use_tracking_in_composition=None,
                 predict_use_cell=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 classifier_keep_rate=None,
                 context_args=None,
                 composition_args=None,
                 with_attention=False,
                 data_type=None,
                 target_vocabulary=None,
                 onmt_module=None,
                 flags=None,
                 data_manager=None,
                 **kwargs
                 ):
        super(NMTModel, self).__init__()
        self.model_type=flags.model_type
        if self.model_type=="SPINN":
            encoder_builder=spinn_builder
        else:
            encoder_builder=rnn_builder
        self.encoder=encoder_builder(
            model_dim=model_dim,
            word_embedding_dim=word_embedding_dim,
            vocab_size=vocab_size,
            initial_embeddings=initial_embeddings,
            fine_tune_loaded_embeddings=fine_tune_loaded_embeddings,
            num_classes=num_classes,
            embedding_keep_rate=embedding_keep_rate,
            tracking_lstm_hidden_dim=tracking_lstm_hidden_dim,
            transition_weight=transition_weight,
            use_sentence_pair=use_sentence_pair,
            lateral_tracking=lateral_tracking,
            tracking_ln=tracking_ln,
            use_tracking_in_composition=use_tracking_in_composition,
            predict_use_cell=predict_use_cell,
            use_difference_feature=use_difference_feature,
            use_product_feature=use_product_feature,
            classifier_keep_rate=classifier_keep_rate,
            mlp_dim=mlp_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_ln=mlp_ln,
            context_args=context_args,
            composition_args=composition_args,
            with_attention=with_attention,
            data_type=data_type,
            onmt_module=onmt_module,
            FLAGS=flags,
            data_manager=data_manager
        )
        assert not (
            use_tracking_in_composition and not lateral_tracking), "Lateral tracking must be on to use tracking in composition."
        self.model_dim = model_dim
        self.data_type=data_type
        sys.path.append(onmt_module)
        from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder, RNNDecoderBase
        from onmt.encoders.rnn_encoder import RNNEncoder
        from onmt.modules import Embeddings
        self.output_embeddings=Embeddings(self.model_dim, len(target_vocabulary)+1, 0)
        self.decoder=StdRNNDecoder("LSTM", False, 1,self.model_dim, embeddings=self.output_embeddings)
        self.target_vocabulary=target_vocabulary
        self.generator=nn.Sequential(
                nn.Linear(self.model_dim, len(self.target_vocabulary)+1),
                nn.LogSoftmax()
            )


    def forward(
            self,
            sentences,
            transitions,
            y_batch=None,
            use_internal_parser=False,
            validate_transitions=True,
            **kwargs):
        example, spinn_outp, attended, transition_loss, transitions_acc=self.encoder(sentences, transitions, y_batch, use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
        nfeat=1#5984#self.output_embeddings.embedding_size
        target_maxlen=max([len(x) for x in y_batch])
        maxlen= example.tokens.size()[1]#max([len(x) for x in attended])
        tmp_trg=[]
        t_mask=[]
        for x in y_batch:
            arr=np.array(list(x)+[1]*(target_maxlen-len(x)))
            t_mask.append([1]*(len(x)+1)+[0]*(target_maxlen-len(x)))
            #arr=x+[1]*(target_maxlen-len(x))
            tmp=[]
            for y in arr:
                la=y
                tmp.append(la)
            tmp_trg.append(tmp)
        trg=[]
        batch_size=example.tokens.size()[0]
        t_tmask_trg=[]
        for i in range(target_maxlen):
            tmp=[]
            tmp_mask=[]
            for j in range(batch_size):
                tmp.append(tmp_trg[j][i])
                tmp_mask.append(t_mask[j][i])
            trg.append(tmp)
            t_tmask_trg.append(tmp_mask)
        actual_dim=spinn_outp[0].shape[-1]
        enc_output=spinn_outp[0].view(1,batch_size, actual_dim)
        padded_enc_output=to_gpu(torch.zeros((1, batch_size, self.model_dim)))
        padded_enc_output[:,:,:actual_dim]=enc_output
        trg=torch.tensor(np.array(trg)).view((target_maxlen, batch_size,nfeat)).long()
        trg=to_gpu(Variable(trg, requires_grad=False))
        if self.model_type=="SPINN":
            src=torch.cat([torch.cat(x[::-1]).unsqueeze(0) for x in example.bufs]).transpose(0,1)  
        else:
            src=example.bufs         
            attended=attended.transpose(0,1)
        enc_state=self.decoder.init_decoder_state(src, attended, (padded_enc_output, padded_enc_output))
        target_forced=False;padded_enc_output=None;enc_output=None; t_mask=None; tmp_trg=None
        if self.training:
            if target_forced:
                decoder=self.decoder(trg, attended, enc_state)
                output=self.generator(decoder[0])
            else:
                unk_token=to_gpu(Variable(torch.zeros((1, batch_size, 1))), requires_grad=False).long()
                inp=unk_token
                dec_state=enc_state
                output=[]
                for i in range(target_maxlen+1):
                    if i==0:
                        inp=unk_token
                    else:
                        inp=trg[i-1].unsqueeze(0)
                    dec_out, dec_state, attn=self.decoder(inp, attended, dec_state, step=i)
                    output.append(self.generator(dec_out.squeeze(0)).unsqueeze(0))
                output=torch.cat(output)
        else:#now just predict:
            unk_token=to_gpu(Variable(torch.zeros((1, batch_size, 1))), requires_grad=False).long()
            inp=unk_token
            maxpossible=100
            dec_state=enc_state
            predicted=[]
            #TODO: replace with k-beam search
            #inp= trg[0].unsqueeze(0)
            for i in range(100):
                dec_out, dec_state, attn = self.decoder(inp, attended, dec_state, step=i)
                out=self.generator(dec_out.squeeze(0))
                argmaxed=torch.max(out,1)[1]
                inp=argmaxed.unsqueeze(1).unsqueeze(0)
                predicted.append(argmaxed)
            return predicted
        return output, trg, None, torch.tensor(t_tmask_trg)


    
