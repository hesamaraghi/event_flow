import torch
import matplotlib.pyplot as plt
from loss.flow import  AEE

class AEE_HIST(AEE):
    
    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)
        self._masked_error_list = None
        self._masked_gtflow_list = None
        
    def reset_all(self):
        super().reset()
        self._masked_error_list = None
        self._masked_gtflow_list = None
        
    def mask_for_flow(self):
        event_mask = self._event_mask[:, -1, :, :].bool()
        gtflow_mask_x = self._gtflow[:, 0, :, :] == 0.0
        gtflow_mask_y = self._gtflow[:, 1, :, :] == 0.0
        gtflow_mask = gtflow_mask_x | gtflow_mask_y
        gtflow_mask = ~gtflow_mask
        mask = event_mask & gtflow_mask
        return mask.squeeze()
    
    def masked_optical_flow(self,flow, mask):
        masked_flow = flow[:,:,mask]
        masked_gtflow = self._gtflow[:,:,mask]
        masked_error = masked_flow - masked_gtflow
        return masked_error.detach().cpu(), masked_gtflow.detach().cpu()
    
    def flow_accumulation(self):
        flow = self._flow_map[-1] * self.flow_scaling
        flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)
        mask = self.mask_for_flow()
        masked_error, masked_gtflow = self.masked_optical_flow(flow, mask)
        if self._masked_error_list is None:
            self._masked_error_list = [masked_error]
        else:
            self._masked_error_list.append(masked_error)
        if self._masked_gtflow_list is None:
            self._masked_gtflow_list = [masked_gtflow]
        else:
            self._masked_gtflow_list.append(masked_gtflow)
    
    def optical_flow_vec2scalar(self, flow, reduce_norm='L2'):
        flow = torch.cat(flow, dim=2)
        if reduce_norm == 'L1':
            return torch.sum(torch.abs(flow), dim=1)
        elif reduce_norm == 'L2':
            return torch.sqrt(torch.sum(flow**2, dim=1))
            
    def calculate_error_hist(self,n_bins=10):

        gtflow_mag = self.optical_flow_vec2scalar(self._masked_gtflow_list).view(-1)
        error_mag = self.optical_flow_vec2scalar(self._masked_error_list).view(-1)
        # Compute histogram and bin centers
        hist, bin_edges = torch.histogram(gtflow_mag, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute mean and standard deviation for each bin
        mean_values = []
        std_values = []
        for i in range(n_bins):
            idx = torch.where((gtflow_mag >= bin_edges[i]) & (gtflow_mag < bin_edges[i+1]))
            mean_values.append(torch.mean(error_mag[idx]))
            std_values.append(torch.std(error_mag[idx]))
            
        # Create plot
        fig, ax1 = plt.subplots()
        ax1.bar(bin_centers, hist/gtflow_mag.shape[0], width=bin_centers[1]-bin_centers[0], color='gray', alpha=0.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('histogram of GT flow magnitude')
        ax2 = ax1.twinx()
        ax2.errorbar(bin_centers, mean_values, yerr=std_values, fmt='o', color='black')
        ax2.set_ylabel('AEE')
        ax2.set_title('Histogram with Mean and Standard Deviation')
        plt.show()