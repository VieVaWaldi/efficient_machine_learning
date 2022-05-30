import torch


class Tester:

    def test(self, test_dataloader, model, loss_fn):

        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)

        # Set model to normal mode
        model.eval()

        test_loss, correct = 0, 0

        for x, Y in test_dataloader:

            # Make a prediction
            prediction = model.forward(x)

            # Calculate real loss
            test_loss += loss_fn(prediction, Y).item()
            correct += (prediction.argmax(1) ==
                        Y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        
        info = f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        print(info)
        return info

