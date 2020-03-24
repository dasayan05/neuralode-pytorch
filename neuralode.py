import torch

class ODELayerFunc(torch.autograd.Function):
        
    @staticmethod
    def forward(context, z0, t_range_forward, dynamics, *theta):
        delta_t = t_range_forward[1] - t_range_forward[0] # get the step size
        
        zt = z0.clone()
        for tf in t_range_forward: # Forward eular's method
            f = dynamics(zt, tf)
            zt = zt + delta_t * f # update

        context.save_for_backward(zt, t_range_forward, delta_t, *theta)
        context.dynamics = dynamics # 'save_for_backwards() won't take it, so..
        
        return zt # final evaluation of 'zt', i.e., zT

    @staticmethod
    def backward(context, adj_end):
        # Unpack the stuff saved in forward pass
        zT, t_range_forward, delta_t, *theta = context.saved_tensors
        dynamics = context.dynamics
        t_range_backward = torch.flip(t_range_forward, [0,]) # Time runs backward

        zt = zT.clone().requires_grad_()
        adjoint = adj_end.clone()
        dLdp = [torch.zeros_like(p) for p in theta] # Parameter grads (an accumulator)

        for tb in t_range_backward:
            with torch.set_grad_enabled(True):
                # above 'set_grad_enabled()' is required for the graph to be created ...
                f = dynamics(zt, tb)
                # ... and be able to compute all vector-jacobian products
                adjoint_dynamics, *dldp_ = torch.autograd.grad([-f], [zt, *theta], grad_outputs=[adjoint])
            
            for i, p in enumerate(dldp_):
                dLdp[i] = dLdp[i] - delta_t * p # update param grads
            adjoint = adjoint - delta_t * adjoint_dynamics # update the adjoint
            zt.data = zt.data - delta_t * f.data # Forward eular's (backward in time)

        return (adjoint, None, None, *dLdp)

class ODELayer(torch.nn.Module):

    def __init__(self, dynamics, t_start = 0., t_end = 1., granularity = 25):
        super().__init__()

        self.dynamics = dynamics
        self.t_start, self.t_end, self.granularity = t_start, t_end, granularity
        self.t_range = torch.linspace(self.t_start, self.t_end, self.granularity)

    def forward(self, input):
        return ODELayerFunc.apply(input, self.t_range, self.dynamics, *self.dynamics.parameters())