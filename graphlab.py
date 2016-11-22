import graphlab

users = []
items = []
rates = []

f = open("./bytecup2016data/invited_info_train.txt","r") 
line_file = f.readline().strip().split("\t")
while line_file and len(line_file) > 0 :
    
    items.append(line_file[0])
    users.append(line_file[1])
    rates.append(line_file[2])

    line_file = f.readline().strip().split("\t")
     

train = graphlab.SFrame({'user_id':  users, 'item_id': items, 'target': rates })

model = graphlab.ranking_factorization_recommender.create(train)

users = []
items = []
rates = []

f = open("./bytecup2016data/test_nolabel.txt","r") 
line_file = f.readline().strip().split(",")
line_file = f.readline().strip().split(",")

while line_file and len(line_file) > 0 :
    question = line_file[0]
    user = line_file[1]
    
    users.append(user)
    items.append(question)

    line_file = f.readline().strip().split(",")


test = graphlab.SFrame({'user_id':  users, 'item_id': items})

rates = model.predict(test)

f = open("./bytecup2016data/test_graph_pos.csv", "w") 
for user, item, label in zip(users, items, rates):
    f.write(item + "," + user + "," + str(label) + "\n" )   

