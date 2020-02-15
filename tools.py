import torch


def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def jacobian(outputs, inputs, create_graph=False):
    '''
    Compute the Jacobian of `outputs` with respect to `inputs`
    jacobian(x, x)
    jacobian(x * y, [x, y])
    jacobian([x * y, x.sqrt()], [x, y])
    '''
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out


def eval_hessian(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()



def KL_diag_gaussian(mu_1,diag_1,mu_2,diag_2):
    ratio=diag_1/diag_2
    return torch.sum(0.5*(mu_1-mu_2)**2/diag_2)+0.5*torch.sum(ratio)-0.5*torch.sum(torch.log(ratio))-mu_1.size(0)/2


def low_rank_cov_inverse(L,sigma):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    var=sigma**2
    inverse_var=1.0/var
    inner_inverse=torch.inverse(torch.diag(torch.ones([rank]))+inverse_var*(L.t())@L)
    return inverse_var*torch.diag(torch.ones([dim]))-inverse_var**2*L@inner_inverse@L.t()

def low_rank_gaussian_logdet(L,sigma):
    dim=L.size(0)
    rank=L.size(1)
    var=sigma**2
    inverse_var=1.0/var
    return torch.logdet(torch.diag(torch.ones([rank]))+inverse_var*(L.t())@L)+dim*tf.log(var)


def KL_low_rank_gaussian_with_diag_gaussian(mu_1,L_1,sigma_1,mu_2,sigma_2,cuda=True):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    if cuda:
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]).cuda())+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]).cuda())*var_1
    else:
        logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
        cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1
    mu_diff=(mu_1-mu_2).view(-1,1)
    var_2=sigma_2**2
    return -0.5*(logdet_1-dim_1*torch.log(var_2)+dim_1-(1/var_2)*torch.trace(cov_1)-(1/var_2)*mu_diff.t()@mu_diff)

def KL_low_rank_gaussian_with_low_rank_gaussian_cuda(mu_1,L_1,sigma_1,mu_2,L_2,sigma_2):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]).cuda())+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
    cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]).cuda())*var_1


    dim_2=L_2.size(0)
    rank_2=L_2.size(1)
    var_2=sigma_2**2
    inverse_var_2=1.0/var_2
    logdet_2=torch.logdet(torch.diag(torch.ones([rank_2]).cuda())+inverse_var_2*(L_2.t())@L_2)+dim_1*torch.log(var_2)

    inner_inverse_2=torch.inverse(torch.diag(torch.ones([rank_2]).cuda())+inverse_var_2*(L_2.t())@L_2)
    cov_inverse_2=inverse_var_2*torch.diag(torch.ones([dim_2]).cuda())-inverse_var_2**2*L_2@inner_inverse_2@L_2.t()

    mu_diff=(mu_1-mu_2).view(-1,1)
    return -0.5*(logdet_1-logdet_2+dim_1-torch.trace(cov_1@cov_inverse_2)-mu_diff.t()@ cov_inverse_2@mu_diff)


def KL_low_rank_gaussian_with_low_rank_gaussian(mu_1,L_1,sigma_1,mu_2,L_2,sigma_2):
    dim_1=L_1.size(0)
    rank_1=L_1.size(1)
    var_1=sigma_1**2
    inverse_var_1=1.0/var_1
    logdet_1=torch.logdet(torch.diag(torch.ones([rank_1]))+inverse_var_1*(L_1.t())@L_1)+dim_1*torch.log(var_1)
    cov_1=L_1@L_1.t()+torch.diag(torch.ones([dim_1]))*var_1


    dim_2=L_2.size(0)
    rank_2=L_2.size(1)
    var_2=sigma_2**2
    inverse_var_2=1.0/var_2
    logdet_2=torch.logdet(torch.diag(torch.ones([rank_2]))+inverse_var_2*(L_2.t())@L_2)+dim_1*torch.log(var_2)

    inner_inverse_2=torch.inverse(torch.diag(torch.ones([rank_2]))+inverse_var_2*(L_2.t())@L_2)
    cov_inverse_2=inverse_var_2*torch.diag(torch.ones([dim_2]))-inverse_var_2**2*L_2@inner_inverse_2@L_2.t()

    mu_diff=(mu_1-mu_2).view(-1,1)
    return -0.5*(logdet_1-logdet_2+dim_1-torch.trace(cov_1@cov_inverse_2)-mu_diff.t()@ cov_inverse_2@mu_diff)


def general_kl_divergence(mu_1,cov_1,mu_2,cov_2):
    mu_diff=(mu_1-mu_2).view(-1,1)
    cov_2_inverse=torch.inverse(cov_2)
    return -0.5*(torch.logdet(cov_1)-torch.logdet(cov_2)+mu_1.size(0)-torch.trace(cov_1@cov_2_inverse)-mu_diff.t()@cov_2_inverse@mu_diff)

def low_rank_gaussian_one_sample(mu,L,sigma,cuda=True):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    if cuda:
        eps_z=torch.randn([rank]).cuda()
        eps=torch.randn([dim]).cuda()
    else:
        eps_z=torch.randn([rank])
        eps=torch.randn([dim])

    return eps_z@L.t()+eps*sigma+mu

def low_rank_gaussian_sample(mu,L,sigma,amount,cuda=True):
    # L is D*R
    dim=L.size(0)
    rank=L.size(1)
    if cuda:
        eps_z=torch.randn([amount,rank]).cuda()
        eps=torch.randn([amount,dim]).cuda()
    else:
        eps_z=torch.randn([amount,rank])
        eps=torch.randn([amount,dim])

    return eps_z@L.t()+eps*sigma+mu


def sample_from_batch_categorical(batch_logits,cuda=True):
    ### shape batch*dim
    ### gumbel max trick
    if cuda:
        noise = torch.rand(batch_logits.size()).cuda()
    else:
        noise = torch.rand(batch_logits.size())
    return torch.argmax(batch_logits - torch.log(-torch.log(noise)), dim=-1)


def sample_from_batch_categorical_multiple(batch_logits,sample_num,cuda=True):
    ### shape batch*dim
    ### gumbel max trick
    shape=list(batch_logits.size())
    shape.insert(-1, sample_num)
    if cuda:
        noise = torch.rand(shape).cuda()
    else:
        noise = torch.rand(shape)
    batch_logits_multiple=batch_logits.repeat(1,1,1,sample_num).view(shape)
    return torch.argmax(batch_logits_multiple - torch.log(-torch.log(noise)), dim=-1)


def one_hot_embedding(labels, num_classes,cuda=True):
    if cuda:
        y = torch.eye(num_classes).cuda()
    else:
        y = torch.eye(num_classes)
    return y[labels]
