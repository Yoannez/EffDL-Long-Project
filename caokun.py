import torch
import torch.nn as nn

class FP():
    def __init__(self, model):
        self.LoadModel(model)

    def LoadModel(self, model):
        convCnt = 0
        self.conv_index_dict = {}
        self.target_modules = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                if isinstance(m, nn.Conv2d):
                    self.conv_index_dict[convCnt] = len(self.target_modules)
                    convCnt += 1
                self.target_modules.append(m)
            elif isinstance(m, nn.Linear):
                for layer in reversed(self.target_modules): # We will not prune the layer next to the linear layer
                    if isinstance(layer, nn.Conv2d):
                        convCnt -= 1
                        del self.conv_index_dict[len(self.conv_index_dict)-1]
                        break
                    self.target_modules.pop()                          
        self.prune_rate = torch.zeros(convCnt)

    def GetPrunableConvs(self):
        convs = []
        for key in self.conv_index_dict:
            convs.append(self.target_modules[self.conv_index_dict[key]])
        return convs

    def SetPruneRate(self, prune_rate_list):
        if len(prune_rate_list) != len(self.conv_index_dict):
            raise ValueError('The length of input list({}) should be the number of prunable conv layers({})'.format(len(prune_rate_list), len(self.conv_index_dict)))
        self.prune_rate = prune_rate_list
    
    def PruneByIndex(self, index, *rate):
        if rate == None:
            rate = self.prune_rate[index]
        self._prune(self.conv_index_dict[index], rate[0])

    def PrunePart(self, index_list, *rate_list):
        rate_list = rate_list[0]
        if rate_list == None:
            for idx in index_list:
                self.PruneByIndex(idx)
        else:
            if len(index_list) != len(rate_list):
                print(len(index_list))
                print(len(rate_list))
                raise ValueError("The two input lists does not match in length")
            for i, idx in enumerate(index_list):
                self.PruneByIndex(idx, rate_list[i])

    def PruneAll(self):
        for i in range(len(self.conv_index_dict)):
            self._prune(self.conv_index_dict[i], self.prune_rate[i])

    def _prune(self, index, rate):
        target = self.target_modules[index]
        len_targets = len(self.target_modules)
        if isinstance(target, nn.Conv2d):
            # Prune the current layer
            nparams_toprune = _compute_nparams_toprune(rate, target.out_channels)
            channels_toprune = self._get_channels_toprune(target.weight.data, 0, nparams_toprune)
            self._prune_conv(target, 0, channels_toprune)

            # Prune the next layer affected
            if index+1 >= len_targets:
                print("Warning: unexpected final layer of type Conv2d")
                return
            target_next = self.target_modules[index+1]
            if isinstance(target_next, nn.BatchNorm2d):
                print('dim=1 BatchNorm\nBefore: ', list(target_next.weight.data.size()))
                self._prune_norm(target_next, channels_toprune)
                print('After: ', list(target_next.weight.data.size()))

                if index+2 >= len_targets:
                    print("Warning: unexpected final layer of type BatchNorm2d")
                    return
                target_next = self.target_modules[index+2]
                if isinstance(target_next, nn.Conv2d):
                    self._prune_conv(target_next, 1, channels_toprune)
                else:
                    print("Warning: unexpected layer type after BatchNorm2d")
        elif isinstance(self.target_modules[index], nn.BatchNorm2d):
            print('next index')
    
    def _get_channels_toprune(self, kernel, dim, nparams_toprune):
        norm = _compute_norm(kernel, 2, dim)
        _, indices = torch.topk(norm, k=nparams_toprune, largest=False)
        return indices

    def _prune_conv(self, conv, dim, channels_toprune):
        channels_tokeep = _get_channels_tokeep(conv.weight.data, dim, channels_toprune)
        if dim == 0:
            print('dim=0 Conv\nBefore: ', list(conv.weight.data.size()))
            conv.weight = nn.Parameter(torch.index_select(conv.weight.data, dim, channels_tokeep))
            if conv.bias != None:
                conv.bias = nn.Parameter(torch.index_select(conv.bias.data, 0, channels_tokeep))
            conv.out_channels = len(channels_tokeep)
            print('After: ', list(conv.weight.size()))

        elif dim == 1:
            print('dim=1 Conv\nBefore: ', list(conv.weight.data.size()))
            conv.weight = nn.Parameter(torch.index_select(conv.weight.data, dim, channels_tokeep))
            conv.in_channels = len(channels_tokeep)
            print('After: ', list(conv.weight.size()))
    
    def _prune_norm(self, norm, channels_toprune):
        print('Norm\nBefore: ', list(norm.weight.data.size()))
        channels_tokeep = _get_channels_tokeep(norm.weight.data, 0, channels_toprune)
        norm.weight.data = torch.index_select(norm.weight.data, 0, channels_tokeep)
        norm.bias.data = torch.index_select(norm.bias.data, 0, channels_tokeep)
        if norm.track_running_stats:
            norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, channels_tokeep)
            norm.running_var.data = torch.index_select(norm.running_var.data, 0, channels_tokeep)
        norm.num_features = len(list(channels_tokeep))
        print('After: ', list(norm.weight.data.size()))

def _compute_nparams_toprune(amount, tensor_size):
    if type(amount) == int:
        return amount
    else:
        return round(amount * tensor_size)

def _compute_norm(t, n, dim):
    # dims = all axes, except for the one identified by `dim`
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    norm = torch.norm(t, p=n, dim=dims)
    return norm

def _get_channels_tokeep(t, dim, channels_toprune):
    num_channels = t.size(dim)
    return torch.LongTensor(list(set(range(num_channels))-set(channels_toprune.tolist())))

class FP_DenseNet(FP):
    def LoadModel(self, model):
        convCnt = 0
        self.conv_index_dict = {}
        self.target_modules = []
        for named_mod in model.named_modules():
            if isinstance(named_mod[1], nn.Conv2d) or isinstance(named_mod[1], nn.BatchNorm2d) or isinstance(named_mod[1], nn.Linear):
                if isinstance(named_mod[1], nn.Conv2d):
                    self.conv_index_dict[convCnt] = len(self.target_modules)
                    convCnt += 1
                self.target_modules.append(named_mod)
            # elif isinstance(named_mod[1], nn.Linear):
            #     for named_mod in reversed(self.target_modules): # We will not prune the layer next to the linear layer
            #         if isinstance(named_mod[1], nn.Conv2d):
            #             convCnt -= 1
            #             del self.conv_index_dict[len(self.conv_index_dict)-1]
            #             break
            #         self.target_modules.pop()                  
        self.prune_rate = torch.zeros(convCnt)
        self.growth_rate = model.growth_rate

    def _prune(self, index, rate):
        # if 'dense1' in self.target_modules[index][0]:
        #     return
        # if 'conv1' in self.target_modules[index][0]:
        #     return
        # if 'dense4' in self.target_modules[index][0]:
        #     return
        # if 'trans3.conv' in self.target_modules[index][0]:
        #     return

        target = self.target_modules[index][1]
        num_targets = len(self.target_modules)
        if isinstance(target, nn.Conv2d):
            # Prune the current layer
            nparams_toprune = _compute_nparams_toprune(rate, target.out_channels)
            channels_toprune = self._get_channels_toprune(target.weight.data, 0, nparams_toprune)
            self._prune_conv(target, 0, channels_toprune)

            # Prune the next layer affected
            if index+1 >= num_targets:
                print("Warning: unexpected final layer of type Conv2d")
                return
            target_next = self.target_modules[index+1][1]
            target_next_name = self.target_modules[index+1][0]
            if isinstance(target_next, nn.BatchNorm2d):
                self._prune_norm(target_next, channels_toprune)

                # Prune all dependenct layers
                if 'bn1' in target_next_name:
                    nblock = int(target_next_name.split('.')[0][-1])
                    offset = 0
                    print(channels_toprune)
                    for i in range(index+2, num_targets):
                        named_mod =  self.target_modules[i]
                        if named_mod[0]=='linear':
                            continue
                        pos = named_mod[0].split('.')
                        if named_mod[0]=='bn' or int(pos[0][-1]) == nblock:
                            if 'bn1' in named_mod[0] or ('t' in pos[0] and pos[1]=='bn') or named_mod[0]=='bn': 
                                print('Dependecy: {}'.format(named_mod[0]))
                                offset += self.target_modules[i-1][1].out_channels
                                channels_toprune_offset = [idx+offset for idx in channels_toprune]
                                channels_toprune_mod = torch.LongTensor(channels_toprune_offset)
                                print(channels_toprune_mod)
                                self._prune_norm(named_mod[1], channels_toprune_mod)
                                last = self.target_modules[i+1][1]
                                if isinstance(last, nn.Conv2d):
                                    self._prune_conv(self.target_modules[i+1][1], 1, channels_toprune_mod)
                                elif isinstance(last, nn.Linear):
                                    self._prune_linear_in_channels(self.target_modules[i+1][1], channels_toprune_mod)                        
                        else:
                            break

                if index+2 >= num_targets:
                    print('Warning: unexpected final layer of type BatchNorm2d')
                    return
                target_next = self.target_modules[index+2][1]
                if isinstance(target_next, nn.Conv2d):
                    self._prune_conv(target_next, 1, channels_toprune)
                elif isinstance(target_next, nn.Linear):
                    self._prune_linear_in_channels(target_next, channels_toprune)
                else:
                    print('Warning: unexpected layer type after BatchNorm2d')
        elif isinstance(self.target_modules[index], nn.BatchNorm2d):
            print('next index')
    
    def _prune_linear_in_channels(self, linear, channels_toprune):
        channels_tokeep = _get_channels_tokeep(linear.weight.data, 1, channels_toprune)
        linear.weight.data = torch.index_select(linear.weight.data, 1, channels_tokeep)
        linear.in_features = len(list(channels_tokeep))
