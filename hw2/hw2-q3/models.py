import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Linear layers for encoder hidden states (Wh) and decoder hidden state (Ws)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Scoring vector (v)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        # Output layer for context and query combination
        self.W_out = nn.Linear(hidden_size*2, hidden_size, bias=False)

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """
        # Extract dimensions
        _, tgt_len, _ = query.size()
        _, src_len, _ = encoder_outputs.size()

        # Expand query to match encoder outputs
        query_expanded = query.unsqueeze(2).expand(-1, -1, src_len, -1)  # (batch, tgt_len, src_len, hidden_size)
        encoder_expanded = encoder_outputs.unsqueeze(1).expand(-1, tgt_len, -1, -1)  # (batch, tgt_len, src_len, hidden_size)

        # Compute alignment scores
        score = self.v(
            torch.tanh(
                self.W_s(query_expanded) + self.W_h(encoder_expanded)
            )
        ).squeeze(-1)  # (batch, tgt_len, src_len)

        # Apply masking for padding in encoder_outputs
        mask = self.sequence_mask(src_lengths).unsqueeze(1)  # (batch, 1, src_len)
        score = score.masked_fill(~mask, float('-inf'))

        # Calculate attention weights
        alignment = F.softmax(score, dim=-1)  # (batch, tgt_len, src_len)

        # Compute context vector
        context = torch.bmm(alignment, encoder_outputs)  # (batch, tgt_len, hidden_size)

        # Concatenate context and query, then pass through Linear transformation
        attn_out = torch.tanh(self.W_out(torch.cat([context, query], dim=-1)))  # (batch, tgt_len, hidden_size)

        return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # Implemented the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################

        # embed the source sequences and apply dropout
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        
        # pack the embeddings and pass through LSTM
        packed_embedded = pack(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, final_hidden = self.lstm(packed_embedded)

        # unpack the embeddings and apply dropout
        enc_output, _ = unpack(packed_output, batch_first=True)
        enc_output = self.dropout(enc_output)

        return enc_output, final_hidden
        
        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # Implemented the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(output, encoder_outputs, src_lengths)
        #############################################

        # embed the target sequences and apply dropout
        embedded = self.embedding(tgt)
        embedded = self.dropout(embedded)
        
        outputs = []
        
        # loop over each time step
        for t in range(tgt.size(1)):
            # get the input for the current time step and pass through LSTM
            input_t = embedded[:, t, :].unsqueeze(1)
            output, dec_state = self.lstm(input_t, dec_state)
            
            if self.attn is not None:
                output = self.attn(output, encoder_outputs, src_lengths)
            
            output = self.dropout(output)
            outputs.append(output)
        
        # Concatenate outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)

        return outputs, dec_state
        
        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt[:, :-1], dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
