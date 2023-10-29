import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


def build_model(conf):
    if conf.family == "gpt2_nn":
        model = TransformerNN(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            hidden_layer_size=conf.hidden_layer_size,
            n_in_intermediate=conf.n_in_intermediate,
            n_out_intermediate=conf.n_out_intermediate,
        )
    else:
        raise NotImplementedError

    return model


class TransformerNN(nn.Module):
    def __init__(
        self, 
        n_dims, 
        n_positions, 
        n_embd=128, 
        n_layer=12, 
        n_head=4, 
        hidden_layer_size=4, 
        n_in_intermediate=0, 
        n_out_intermediate=0,
        hidden_sep_embed=False,
    ):
        super(TransformerNN, self).__init__()
        configuration = GPT2Config(
            n_positions=(2 + n_in_intermediate) * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_nn_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_in_intermediate = n_in_intermediate
        self.n_out_intermediate = min(n_in_intermediate, n_out_intermediate)

        self._read_in_x = nn.Linear(n_dims, n_embd)
        self._read_in_y = nn.Linear(1, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

        # if hidden_sep_embed is True: different intermediate inputs/outputs use different embedding layers
        # else: all the intermediate inputs/outputs use the same embedding layer
        self.hidden_sep_embed = hidden_sep_embed
        if hidden_sep_embed:
            self._read_in_hidden = nn.ModuleList(
                [nn.Linear(hidden_layer_size, n_embd) for i in range(self.n_in_intermediate)]
            )
            self._read_out_hidden = nn.ModuleList(
                [nn.Linear(n_embd, hidden_layer_size) for i in range(self.n_out_intermediate)]
            )
        else:
            if self.n_in_intermediate > 0:
                self._read_in_hidden = nn.Linear(hidden_layer_size, n_embd)
            if self.n_out_intermediate > 0:
                self._read_out_hidden = nn.Linear(n_embd, hidden_layer_size)

    def _combine_embed(self, xs_b, ys_b, layer_activations=None):
        bsize, points, dim = xs_b.shape
        xs_embed = self._read_in_x(xs_b)
        stacked_tensors = [xs_embed]

        if self.n_in_intermediate > 0:
            ss_embeds = []
            for i in range(self.n_in_intermediate):
                act = layer_activations[i]
                if self.hidden_sep_embed:
                    ss_embeds.append(self._read_in_hidden[i](act))
                else:
                    ss_embeds.append(self._read_in_hidden(act))
            stacked_tensors += ss_embeds
        
        ys_embed = self._read_in_y(ys_b.reshape(bsize, points, 1))
        stacked_tensors += [ys_embed]

        zs_embed = torch.stack(stacked_tensors, dim=2)
        zs_embed = zs_embed.view(bsize, len(stacked_tensors) * points, self.n_embd)
        return zs_embed

    def forward(self, xs, ys, loss_func, layer_activations=None):
        if ((layer_activations is not None) and (self.n_in_intermediate > len(layer_activations))) \
            or ((layer_activations is None) and (self.n_in_intermediate != 0)):
            raise ValueError("the number of given intermediate features is not consistent with model setting")

        x_idx_freq = self.n_in_intermediate + 2

        embeds = self._combine_embed(xs, ys, layer_activations)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        
        # calculate training loss
        losses = []
        for i in range(self.n_out_intermediate):
            if self.hidden_sep_embed:
                pred_hidden = self._read_out_hidden[i](output[:, i:][:, ::x_idx_freq])
            else:
                pred_hidden = self._read_out_hidden(output[:, i:][:, ::x_idx_freq])
            losses += [loss_func(pred_hidden, layer_activations[i]).sum(-1).mean()]
        pred = self._read_out(output[:, self.n_out_intermediate:][:, ::x_idx_freq])
        losses += [loss_func(pred[:,:,0], ys).mean()]
        return losses, sum(losses)
    
    
    
    def predict(self, xs, ys, layer_activations=None):
        if ((layer_activations is not None) and (self.n_in_intermediate > len(layer_activations))) \
            or ((layer_activations is None) and (self.n_in_intermediate != 0)):
            raise ValueError("the number of given intermediate features is not consistent with model setting")

        x_idx_freq = self.n_in_intermediate + 2

        with torch.no_grad():
            embeds = self._combine_embed(xs, ys, layer_activations)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state

            pred_y = torch.zeros_like(ys)
            for i in range(ys.shape[1]-1, -1, -1):
                xs = xs[:,:i+1]
                ys = ys[:,:i+1]
                if layer_activations is not None:
                    layer_activations = [item[:,:i+1] for item in layer_activations]

                for j in range(self.n_out_intermediate):
                    if self.hidden_sep_embed:
                        layer_activations[j][:,i] = self._read_out_hidden[j](output[:, j:][:, ::x_idx_freq][:,i])
                    else:
                        layer_activations[j][:,i] = self._read_out_hidden(output[:, j:][:, ::x_idx_freq][:,i])
                    embeds = self._combine_embed(xs, ys, layer_activations)
                    output = self._backbone(inputs_embeds=embeds).last_hidden_state
                    
                pred_y[:,i] = self._read_out(output[:, self.n_out_intermediate:][:, ::x_idx_freq][:,i])[:,0]
            return pred_y
    
