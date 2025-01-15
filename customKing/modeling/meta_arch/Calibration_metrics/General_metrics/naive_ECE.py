
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from customKing.modeling.meta_arch.build import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class ECE_with_equal_width(nn.Module):
    def __init__(self,cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True

    def forward(self,y_pred, y_true):
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes = F.softmax(y_pred, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(y_true)

        ece = torch.zeros(1, device=y_pred.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

@META_ARCH_REGISTRY.register()
class ECE_with_equal_mass(nn.Module):
    def __init__(self, cfg, n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.Need_top_confidence = True
        self.plot_name = "$ECE_{bin}$"

    def forward(self,confidences, accuracies):

        mass_in_bin = len(confidences)//self.n_bins
        ece = 0.
        for i in range(self.n_bins):
            Ps = confidences[i*mass_in_bin:(i+1)*mass_in_bin].mean()
            acc_in_bin = accuracies[i*mass_in_bin:(i+1)*mass_in_bin].mean()
            ece += np.abs(Ps - acc_in_bin)/self.n_bins
        return ece


class ECE_binAcc(nn.Module):
    def __init__(self,n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(self,y_pred, y_true):
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes = F.softmax(y_pred, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(y_true)

        Accs = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                Accs.append(accuracy_in_bin)
            else:
                Accs.append(0)

        return Accs

class ECE_binAcc_em(nn.Module):
    def __init__(self,n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(self,confidences, accuracies):
        Accs = []
        bins = []
        mass_in_bin = len(confidences)//self.n_bins
        for i in range(self.n_bins-1):
            acc_in_bin = accuracies[i*mass_in_bin:(i+1)*mass_in_bin].mean()
            Accs.append(acc_in_bin)
            bins.append(confidences[i*mass_in_bin])
        bins.append(confidences[self.n_bins*mass_in_bin-1])
        if bins[0] != 0.:
            bins.insert(0,torch.tensor(0.))
            Accs.insert(0,torch.tensor(0.))
        if bins[-1] != 1.:
            bins.append(torch.tensor(1.))
            Accs.append(Accs[-1])
        return Accs,bins

class TopLabel_Specific_ECE_LOSS_equal_width(nn.Module):
    def __init__(self,n_bins=15) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(self,y_pred, y_true):
        class_labels = np.unique(y_true.detach().cpu().numpy())
        _,top_label = y_pred.max(dim=1)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        eces = []
        for i in class_labels:
            top_label_index = (top_label==i)
            top_label_samples = y_pred[top_label_index]
            top_label_label = y_true[top_label_index]

            softmaxes = F.softmax(top_label_samples, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(top_label_label)

            ece = torch.zeros(1, device=top_label_samples.device)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            eces.append(ece)
        ece = sum(eces)/len(eces)
        return ece 
    