# FEDERATED-LEARNING-THROUGH-DISTRIBUTED-COMPUTING
Parallel and distributed computing Course Project
Mobile computing devices have seen a rapid increase in their computational power as well as storage capacity. Aided by this increased computational power and abundance of data, as well as due to privacy and security concerns, there is a growing trend towards training machine learning models over networks of such devices using only local training data. 

## How federated learning works
Federated Learning is a machine learning setting where the goal is to train a high-quality centralized model with training data distributed over a large number of clients each with unreliable and relatively slow network connections.
It works like this: your device downloads the current model, improves it by learning from data on your phone, and then summarizes the changes as a small focused update. Only this update to the model is sent to the cloud, using encrypted communication, where it is immediately averaged with other user updates to improve the shared model. All the training data remains on your device, and no individual updates are stored in the cloud.
![image](https://user-images.githubusercontent.com/55789995/123841592-527e2600-d942-11eb-9c2a-5d1314abd786.png)

### Python & ML Libraries used
PySyft, an open-source library built for Federate Learning and Privacy Preserving. PySyft allows its users to perform private and secure Deep Learning (DL). It is built as an extension of some DL libraries, such as PyTorch, Keras and Tensorflow.
Other libraries such as: Numpy, panda, sickit learn, PyGrid are used for this project.
Project Components
The final implementation involves the following components:
1.	Client
2.	Gateway
3.	Worker Nodes

The dataset (spamsample.csv) will be broken down into to inputs using the pre-processing script (data_prep.py):
1.	msgtexts.npy – containing the messages converted into numerical data.
2.	msgtypes.npy – containing the message types, whether the message is spam or not.
The values within the msgtypes.npy dataset corresponds to an NumPy array of 30 tokens generated from each text message padded at the left or truncated at right, to obtain a fixed size.
The msgtypes.npy dataset would only have a value of 1 for spam and 0 for non-spam messages.

## Final Implementation
To split up the code and execute it on nodes, we will use PyGrid (https://github.com/OpenMined/PyGrid). It is a framework to extend PySyft into deployable tools that could be used in VMs and Docker containers.
Note: Due to the complexity of deploying to Google Compute Engine (that would be used in the actual demo of the implementation), we will provide simple instructions for deploying to any Linux host or VM in this report. We’ve also removed the need for GPUs to simplify the set up further.

### Requirements:
•	Git

•	Python 3.7+ (installed on your PC and VMs)

•	3 Linux VMs/hosts (1 gateway, 2 worker nodes)

•	1 PC (to connect to gateway and nodes)
### Step 1:
To begin, we will need to clone the git repository into each VM. We will need at least 3 VMs, one for the gateway, two for worker nodes, just like our prototype. To clone the repository execute the following command on each VM:
git clone https://github.com/OpenMined/PyGrid.git
### Step 2:
We then enter the PyGrid directory on each VM and install the perquisites:
cd PyGrid
pip install -r requirements.txt
### Step 3:
On the gateway VM, execute the following:
cd gateway
python gateway.py –start_local_db –port=8080
### Step 4:
Ensure to take down the gate way IP address as you will now need to replace <gateway ip> with the actual IP of the VM.
On worker node 1, execute the following:
cd app/websocket/
python websocket_app.py –start_local_db –id=worker1 –port=3001 –gateway_url=http://<gateway IP>:8080
On worker node 2, execute the following:
cd app/websocket/
python websocket_app.py –start_local_db –id=worker2 –port=3001 –gateway_url=http://<gateway IP>:8080
### Step 5:
Ensure to take down the worker node IP addresses as you will now need to replace the <worker1 IP> and <worker2 IP> with the actual IP of the VMs.
From the submitted project files for this report, execute the following script on your PC. Note that we only upload the data to the nodes, not the gateway:
python upload_data.py -node1 <worker1 IP> -node2 <worker2 IP>
python train_model.py -gw <gateway IP>
After approximately 20mins, you should receive similar output as shown by the prototype.
## Conclusion
In this project we’ve demonstrated that it is possible to train a model with decent accuracy on distributed nodes. The gateway node is not privy to the information stored by the worker nodes, but is able to train and test the neural network model through each node. The nodes were also able to train the model without either node knowing what data the other node has. This implementation serves as an example of a practical way to perform deep learning using distributed nodes in a federated learning setup, for the purpose of detecting spam text messages without revealing the data from users (nodes).  




