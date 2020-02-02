setwd("/Users/tianyudu/Documents/UToronto/Course/ECO499/ugthesis/data/ready_to_use/day_returns")

Mon <- read.table("./Monday.csv", header = TRUE, sep = ",")
Tue <- read.table("./Tuesday.csv", header = TRUE, sep = ",")
Wed <- read.table("./Wednesday.csv", header = TRUE, sep = ",")
Thu <- read.table("./Thursday.csv", header = TRUE, sep = ",")
Fri <- read.table("./Friday.csv", header = TRUE, sep = ",")

t.test(Mon$RETURN, mu=0)
t.test(Tue$RETURN, mu=0)
t.test(Wed$RETURN, mu=0)
t.test(Thu$RETURN, mu=0)
t.test(Fri$RETURN, mu=0)
