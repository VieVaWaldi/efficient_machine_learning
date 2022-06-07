from config import cfg

import torch


class Trainer:

    def train(self, train_dataloader, model, loss_fn, optimizer):
        """
            - batch_size controls the number of training samples before a weight update
            - one epoch is one complete pass through all the data
        """

        # Set model to train mode
        model.train()

        size = len(train_dataloader.dataset)
        loss_total = 0

        num_epochs = cfg["paras"]["epochs"]
        min_avg_loss = cfg["paras"]["min_avg_loss"]
        MODEL_PATH = cfg["paths"]["model"]

        for epoch in range(num_epochs):

            avg_loss = 0
            print(f"Epoch: [{epoch:>3d}|{num_epochs:>3d}]")

            for batch, (x, Y) in enumerate(train_dataloader):

                # Make a prediction
                prediction = model(x)

                # Calculate loss
                loss = loss_fn(prediction, Y)
                avg_loss += loss

                # Backpropagation on loss with the optimizer
                optimizer.zero_grad()
                loss.backward()

                # This should adapt the weights
                optimizer.step()

                if batch % 100 == 0:
                    loss, curr = loss.item(), batch * len(x)
                    print(f"Loss: {loss:.3f} [{curr:>5d}|{size:>5d}]")

            print(
                f"Avg loss: {avg_loss/len(train_dataloader):.3f}, Total loss: {avg_loss:.3f}\n")

            if avg_loss < min_avg_loss:
                print(
                    f"Avergae loss is below {min_avg_loss}, training was stoppped automatically.")
                break

        torch.save(model.state_dict(), MODEL_PATH)

        return loss_total
