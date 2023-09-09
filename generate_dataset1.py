# In[3]:


import random
import csv


high_val = 10

with open('train1.csv', 'w') as f_train, open('valid1.csv', 'w') as f_valid:
    train_writer = csv.writer(f_train)
    valid_writer = csv.writer(f_valid)
    train_writer.writerow(['input', 'label'])
    valid_writer.writerow(['input', 'label'])
    
    for i in range(300000):
        pow1 = random.randint(0,high_val)
        pow2 = random.randint(0,high_val)
        
        num1 = random.choice([-1,1])*random.randint(pow(10,pow1), pow(10,pow1+1))
        num2 = random.choice([-1,1])*random.randint(pow(10,pow2), pow(10,pow2+1))
        input_s = str(num1) + ' + ' + str(num2)
        label = str(num1+num2)
        if i < 299900:
            train_writer.writerow([input_s, label])
        else:
            valid_writer.writerow([input_s, label])
            