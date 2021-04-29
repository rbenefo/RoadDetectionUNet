import numpy as np
from torch.utils.data import Dataset, DataLoader

def train_test(net, criterion, optimizer, 
               train_data_loader, device,
               num_epochs=5, verbose=True, training_plot=True):
  # train
  training_losses = []
  valid_stats = []
  for epoch in tqdm(range(num_epochs)):
    net.train()
    running_loss = []
    for i, data in tqdm(enumerate(train_data_loader)):
      # print(i)
      # get the data
      images, labels, _ = data
      # print("Time {}".format(et-st))
      images = images.to(device).float()
      # kernel = torch.tensor(([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]] * 3] * 3)).to(device).float()
      # images = F.conv2d(images, kernel, padding=1, stride=[1, 1])
      labels = labels.to(device).float()
      # zero the parameter gradients
      optimizer.zero_grad()
      outputs = net(images).squeeze(1)
      # outputs = torch.reshape(outputs, (-1, 256, 256))
      weight = torch.zeros((labels.shape)).to(device)
      weight[torch.where(labels == 1)] = 0.9 #labels ==1, weight_matrix = x
      weight[torch.where(labels == 0)] = 0.1 #labels ==0, weight_matrix = (1-x)
      # weight[torch.where(torch.logical_and(labels == 0, outputs == 1))] = 0.8 #labels ==1, weight_matrix = x

      #DICE coefficient??
      assert weight.shape[0] == labels.shape[0]
      loss = dice_coeff(outputs, labels)
      # loss = loss*weight
      loss = loss.mean()
      loss.backward()
      optimizer.step()
      # print statistics
      training_losses += [loss.item()]
      running_loss.append(loss.item())
    # if epoch % 10 == 0:
    running_loss = np.array(running_loss)
    training_losses += [loss.item()]
    print("Loss {}".format(np.sum(running_loss)/len(running_loss)))
    valid_stats.append(test(net, validation_dataloader))
  if training_plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
    axes[0].plot(training_losses)
    axes[1].plot(valid_stats)
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_ylabel('Training loss')
    axes[1].set_ylabel('IOU Score')
    fig.tight_layout() 
    plt.show()



class MapDataset(Dataset):
  def __init__(self, image_path, mask_path, image_list, mask_list, transforms=None, eval = False):
    self.image_path = image_path
    self.images = image_list
    self.target_path = mask_path
    self.targets = mask_list
    self.transforms = transforms
    self.eval = eval
    print("Init Complete.")
  def __getitem__(self, index):
    # st = time.time()
    image = np.load(os.path.join(self.image_path, self.images[index]))
    #1. Normalize image
    if self.transforms:
      image_t = self.transforms(image)

    mask = np.load(os.path.join(self.target_path, self.targets[index]))
    mask = mask[:,:,0]
    mask[np.where(mask<=128)] = 0
    mask[np.where(mask != 0)] = 1
    #2. If doing augmentation, be sure to apply augmentation to mask as well

    # print("mask binarized")
    # et = time.time()
    # print("Time {}".format(et-st))
    if self.eval:
      print(self.images[index], index)
    return image_t, torch.tensor(mask), image
  def __len__(self):
      return len(self.images)



class TrainTest:
  def __init__(self, image_path, mask_path):
    self.image_path = image_path
    self.mask_path =  mask_path
    self.images = np.array(os.listdir(self.image_path))
    self.targets = np.array(os.listdir(self.mask_path))
    self.randomize()
  
  def randomize(self):
    #create one splitting indices --> apply to both self.images, self.targets
    #split self.images, self.targets via random indices --> 
    size = len(self.images)
    indices = np.random.permutation(np.arange(size).astype(int))
    indices_train = indices[:int(size*0.9)]
    indices_valid = indices[int(size*0.9):]
    self.images_train = self.images[indices_train] #lists of names of the files
    self.images_valid = self.images[indices_valid]
    self.masks_train = self.targets[indices_train]
    self.masks_valid = self.targets[indices_valid]

  def create_datasets(self, transforms =None):
    train_dataset = MapDataset(self.image_path, self.mask_path, self.images_train, self.masks_train, transforms = transforms)
    validation_dataset = MapDataset(self.image_path, self.mask_path, self.images_valid, self.masks_valid, transforms = transforms, eval = True)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers = 4,batch_size=8, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,num_workers = 0, batch_size=8, shuffle=False)
    return train_dataloader, validation_dataloader
