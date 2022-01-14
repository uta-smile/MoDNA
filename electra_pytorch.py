import math
from functools import reduce
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
import os
from transformers import WEIGHTS_NAME

# constants

Results = namedtuple('Results', [
    'loss',
    'mlm_loss',
    'mse_motif'
    'disc_loss',
    'disc_motif_loss',
    'gen_acc',
    'disc_acc',
    'disc_labels',
    'disc_predictions'
])

# helpers

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def prob_mask_like(t, prob):
    '''
    print(t.shape)
    b, length = t.shape
    t1 = torch.zeros(b, int(length/6)).float().uniform_(0,1) 
    t2 = torch.zeros_like(t)<prob


    print(t1.expand(b,t1.shape[1]*6))
    print(t1[0])
    print(t1.shape)
    print(t2[0])
    '''
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

# hidden layer extractor class, for magically adding adapter to language model to be pretrained

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# main electric class 

class Electric(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        *,
        num_tokens = None,
        discr_dim = -1,
        discr_layer = -1,
        mask_prob = 0.15,
        replace_prob = 0.85,
        random_token_prob = 0.,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = [],
        disc_weight = 50.,
        gen_weight = 1.,
        temperature = 1.):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer = discr_layer),
                nn.Linear(discr_dim, 1)
            )

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight
    def forward(self, input, **kwargs):
        
        b, t = input.shape
        
        '''
        replace_prob = prob_mask_like(input, self.replace_prob)
        
        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)
        
        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)
        
        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            random_token_prob = prob_mask_like(input, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input.shape, device=input.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]

        # [mask] input
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)

        '''
        # set inverse of mask to padding tokens for labels
        gen_labels = input.masked_fill(~mask, self.pad_token_id)
        # get generator output and get mlm loss
       
        logits = self.generator(masked_input, **kwargs).logits

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index = self.pad_token_id
        )

        

        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = gumbel_sample(sample_logits, temperature = self.temperature)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()    

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float().detach()


        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(disc_input, **kwargs)
        disc_logits = disc_logits.reshape_as(disc_labels)
       
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        # gather metrics
        with torch.no_grad():
            gen_predictions = torch.argmax(logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            disc_acc = 0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean() + 0.5 * (disc_labels[~mask] == disc_predictions[~mask]).float().mean()

        Output = [self.gen_weight * mlm_loss + self.disc_weight * disc_loss, mlm_loss, mse_motif_loss, disc_loss, gen_acc, disc_acc, disc_labels, disc_predictions]
        
        # return weighted sum of losses
        return Output





# main electra class

class electra(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        *,
        num_tokens = None,
        discr_dim = -1,
        discr_layer = -1,
        mask_prob = 0.15,
        replace_prob = 0.85,
        random_token_prob = 0.,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = [],
        disc_weight = 50.,
        gen_weight = 1.,
        motif_weight = 20.,
        temperature = 1.
        ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer = discr_layer),
                nn.Linear(discr_dim, 1)
            )

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight
        self.motif_weight = motif_weight
        self.mse_loss = torch.nn.MSELoss(reduce = False)
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        # self.kl_loss = torch.nn.KLDivLoss(reduction='sum')
        # self.softmax_layer = torch.nn.Softmax(dim=3)
        self.softmax_layer = torch.nn.LogSoftmax(dim=3)
        self.motif_loss_func = 'kl' # mse/kl
   
    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """

        
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
       # model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        #model_to_save.config.save_pretrained(save_directory)


        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        output_model_file = os.path.join(save_directory)

        #torch.save(model_to_save.state_dict(), output_model_file+'/'+'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)



        #logger.info("Model weights saved in {}".format(output_model_file))

    def forward(self, input, motif_label, motif_mask, motif_l, **kwargs):  #  input: TOKENS / motif_label: PWM matrix / motif_mask 
        
        b, t = input.shape
        
        
        replace_prob = prob_mask_like(input, self.replace_prob)
        
        #print(replace_prob[0])
        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)
        
        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            random_token_prob = prob_mask_like(input, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input.shape, device=input.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]

        # [mask] input
        #print(mask[0])
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)


        # set inverse of mask to padding tokens for labels
        gen_labels = input.masked_fill(~mask, self.pad_token_id)
        # get generator output and get mlm loss

       
        
        #masked_motif = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)
        motif_label = motif_label.clone().detach()
        motif_label = motif_label.to(torch.float32)
        pack_predictions = self.generator(masked_input, motif_label,motif_mask,  **kwargs)
        
        logits = pack_predictions.logits
        logits_motif = pack_predictions.logits_motif
        
        #logits= self.generator(masked_input, motif_label,motif_mask,  **kwargs)[0].logits
        
        
        
        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index = self.pad_token_id
        )
        
        
        #mlm_loss = F.cross_entropy(logits.view(-1,4101 ), gen_labels.view(-1),ignore_index=self.pad_token_id)


        '''
        print("motif_label")
        print(motif_label.shape)
        print("input_id")
        print(input.shape)
        print("gen_labels")
        print(gen_labels.shape)
        print("logits")
        print(logits.transpose(1, 2).shape)

        print(logits.view(-1,4101 ).shape)
        print(gen_labels.view(-1).shape)
        
        print("motif_mask")
        print(motif_mask.shape)
        print("motif")
        print(logits_motif.view(b,-1).shape)
        
        '''

        
        if self.motif_loss_func == 'mse':
            mse_motif_loss = self.mse_loss(logits_motif.view(b,-1), motif_label) # torch.Size([10, 11856])
            mse_motif_loss = torch.mul(mse_motif_loss, motif_mask) 
            mse_motif_loss = mse_motif_loss.mean()

        elif self.motif_loss_func == 'kl':

            tmp_motif_mask = torch.reshape(motif_mask, (b, t, 24))
            tmp_motif_mask = torch.reshape(tmp_motif_mask, (b, t, 6, 4))
            tmp_motif_label = torch.reshape(motif_label, (b, t, 24))
            tmp_motif_label = torch.reshape(tmp_motif_label, (b, t, 6, 4))
            tmp_logits_motif = torch.reshape(logits_motif, (b, t, 6, 4))
            tmp_logits_motif = self.softmax_layer(tmp_logits_motif)
            kl_motif_loss = self.kl_loss(tmp_logits_motif, tmp_motif_label)
            mse_motif_loss = kl_motif_loss
        
        

        # input('debug')
        

        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = gumbel_sample(sample_logits, temperature = self.temperature)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()



        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float().detach()

        motif_l = motif_l.float().detach()
        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(disc_input, **kwargs)
        #disc_motif_logits= self.discriminator(disc_input, **kwargs).motif_logits
        
        disc_logits = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        
        disc_motif_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            motif_l[non_padded_indices]
            )
        
        #mse_motif_loss = 0
        #disc_motif_loss = 0
        # gather metrics
        with torch.no_grad():
            gen_predictions = torch.argmax(logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            disc_acc = 0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean() + 0.5 * (disc_labels[~mask] == disc_predictions[~mask]).float().mean()

        Output = [self.gen_weight * mlm_loss + self.disc_weight * disc_loss + self.motif_weight *mse_motif_loss + self.motif_weight*disc_motif_loss, mlm_loss,mse_motif_loss, disc_loss, disc_motif_loss,gen_acc, disc_acc, disc_labels, disc_predictions]
        #Output = [self.gen_weight * mlm_loss + self.disc_weight * disc_loss, mlm_loss,mse_motif_loss, disc_loss,disc_motif_loss, gen_acc, disc_acc, disc_labels, disc_predictions]
        # return weighted sum of losses
        return Output
