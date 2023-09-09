# In[3]:


import random
import csv


# In[4]:


low_val = -10**10
high_val = 10**10

with open('train_pow100.csv', 'w') as f_train, open('valid_pow100.csv', 'w') as f_valid:
    train_writer = csv.writer(f_train)
    valid_writer = csv.writer(f_valid)
    train_writer.writerow(['input', 'label'])
    valid_writer.writerow(['input', 'label'])
    
    for i in range(500000):
        pow1 = random.randint(0,100)
        pow2 = random.randint(0,100)
        
        num1 = random.choice([-1,1])*random.randint(pow(10,pow1), pow(10,pow1+1))
        num2 = random.choice([-1,1])*random.randint(pow(10,pow2), pow(10,pow2+1))
        input_s = str(num1) + ' + ' + str(num2)
        label = str(num1+num2)
        if i < 499800:
            train_writer.writerow([input_s, label])
        else:
            valid_writer.writerow([input_s, label])
            
