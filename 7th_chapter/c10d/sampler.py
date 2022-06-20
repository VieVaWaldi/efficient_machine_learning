import torch
import torch.utils.data


class SimpleDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 i_length):
        self.m_length = i_length

    def __len__(self):
        return self.m_length

    def __getitem__(self,
                    i_idx):
        return i_idx*10


# 64 Eintraege im Datenatz
l_data_simple = SimpleDataSet(64)
print(l_data_simple)

l_sampler = torch.utils.data.DistributedSampler(
    l_data_simple,
    num_replicas=4,
    rank=0,
    shuffle=False,
    drop_last=False
)

l_data_loader = torch.utils.data.DataLoader(
    l_data_simple, batch_sampler=l_sampler
)

# Aufteilen des datensatzes ueber den distrubted data sampler
# Eintrage gehen bis 630, also 64 eintraege
for l_id, l_x in enumerate(l_data_loader):
    print('id: ', l_id, ' x: ', l_x)
