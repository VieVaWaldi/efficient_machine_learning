# Trains the given MLP-model.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).
#  @param io_model model which is trained.
#  @param io_optimizer.
#  @return summed loss over all training samples.
import torch
import torch.distributed

def print_ol(msg, only_rank_0=False):
    l_rank = torch.distributed.get_rank()
    if only_rank_0:
        if l_rank == 0:
            print(f"Rank #{l_rank}: {msg}")
    else:
        print(f"Rank #{l_rank}: {msg}")

def train(i_loss_func,
          i_size,
          io_data_loader,
          io_model,
          io_optimizer):
    # switch model to training mode
    io_model.train()

    l_loss_total = 0
    size_per_batch = len(io_data_loader.dataset)//torch.distributed.get_world_size()

    print_ol(f"Starting training")
    for l_batch_id, (l_x, l_y) in enumerate(io_data_loader):
        l_prediction = io_model(l_x)
        l_loss = i_loss_func(l_prediction, l_y)
        l_loss_total += l_loss.item()

        # backprop
        io_optimizer.zero_grad()
        l_loss.backward()
        io_optimizer.step()

        # ------------------- 7.3 Addon ------------------- #
        # Gradienten mit All reduce syncen und mitteln

        for l_param in io_model.parameters():
            torch.distributed.all_reduce(l_param.grad.data,
                                            op=torch.distributed.ReduceOp.SUM)
            l_param.grad.data = l_param.grad.data / float(i_size)
        # ------------------------------------------------- #

        if l_batch_id % 100 == 0:
            loss, curr = l_loss.item(), l_batch_id * len(l_x)
            print_ol(f"Loss: {loss:.3f} [{curr:>5d}|{size_per_batch:>5d}]", True)

    return l_loss_total
