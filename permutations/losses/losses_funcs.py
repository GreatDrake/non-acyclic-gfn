import torch 
class SDBLoss:    
    def __call__(self, log_forward, log_backward, log_flow, log_flow_next, loss_mask, args, is_first=True):
        if args.loss_scale:
            inn = torch.abs(log_flow + log_forward - \
                                                log_flow_next - log_backward) ** args.sdb_alpha
            
            mult = (1 + args.sdb_eta * torch.exp(log_flow_next))**args.sdb_beta if is_first else (1 + args.sdb_eta * torch.exp(log_flow))**args.sdb_beta

            reg = args.reg_coef * torch.exp(log_flow_next) if is_first else args.reg_coef * torch.exp(log_flow)

            step_losses = torch.log1p(args.sdb_eps * inn) * mult + reg
        else:
            inn = torch.abs(torch.exp(log_flow + log_forward) - \
                                                torch.exp(log_flow_next + log_backward)) ** args.sdb_alpha
            mult = (1 + args.sdb_eta * torch.exp(log_flow_next))**args.sdb_beta if is_first else (1 + args.sdb_eta * torch.exp(log_flow))**args.sdb_beta
            step_losses = torch.log1p(args.sdb_eps * inn) * mult
        return (step_losses * loss_mask).sum()
    
class DBLoss:
    def __call__(self, log_forward, log_backward, log_flow, log_flow_next, loss_mask, args, is_first=True):
        if args.loss_scale:
            reg = args.reg_coef * torch.exp(log_flow_next) if is_first else args.reg_coef * torch.exp(log_flow)
            inn = (log_flow + log_forward - log_flow_next - log_backward) ** 2 + reg
            step_losses = (inn * loss_mask)
        else:
            forward = torch.exp(log_flow + log_forward)
            backward = torch.exp(log_flow_next + log_backward)
            step_losses = ((forward - backward)**2) * loss_mask
        return step_losses.sum()
    
class GFlowLoss:
    def __init__(self, objective):
        assert objective in ["DB", "SDB"], "Invalid loss"
        if objective == "DB":
            self.func = DBLoss()
        elif objective == "SDB":
            self.func = SDBLoss()
    def __call__(self, log_forward, log_backward, log_flow, log_flow_next, loss_mask, args, is_first):
        return self.func(log_forward, log_backward, log_flow, log_flow_next, loss_mask, args, is_first)