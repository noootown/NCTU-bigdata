library(dplyr)
library(ggplot2)
library(plotly)

setwd('~/Downloads/')
data <- tbl_df(read.table(file="./yellow.csv", header=TRUE, sep=","))

data <- data %>%
  mutate(
    pu_hour = substr(as.character(tpep_pickup_datetime), 12, 13),
    do_hour = substr(as.character(tpep_pickup_datetime), 12, 13),
    distance = ifelse(trip_distance > 15, 1, 0)
    )

# Q1
data %>%
  ggplot(aes(x=pu_hour)) +
  geom_bar(fill = "red", alpha = 0.2) +
  ggtitle("pickup hour vs count") +
  labs(x="hour", y="n")

# Q2
data %>%
  group_by(PULocationID) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

data %>%
  group_by(DOLocationID) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

data %>%
  group_by(PULocationID, DOLocationID) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

#Q3
data %>%
  group_by(distance) %>%
  summarise(
    n= n(),
    avg_passenger = mean(passenger_count),
    avg_fare_amount = mean(fare_amount),
    avg_total_amount = mean(total_amount)
  ) %>%
  View

data %>%
  sample_n(100000) %>%
  ggplot(aes(x=as.factor(distance), y=fare_amount)) +
  geom_point(alpha=0.2, size=1, color="red") +  
  ggtitle("distance vs fare_amount") +
  labs(x="distance", y="fare_amount")

data %>%
  filter(distance == 0) %>%
  ggplot(aes(x=payment_type)) +
  geom_histogram(
    aes(y=(..count..)/sum(..count..) * 100), bins=50, fill = "red", alpha = 0.2) +
  ggtitle("<=15km payment type vs percentage") +
  labs(x="<=15km payment type", y="percentage")

data %>%
  filter(distance == 1) %>%
  ggplot(aes(x=payment_type)) +
  geom_histogram(
    aes(y=(..count..)/sum(..count..) * 100), bins=50, fill = "red", alpha = 0.2) +
  ggtitle(">15km payment type vs percentage") +
  labs(x=">15km payment type", y="percentage")

data %>%
  filter(distance = 0)
  ggplot(aes(x=factor(pu_hour))) +
  geom_bar(aes(y=(..count..)/sum(..count..) * 100), fill = "red", alpha = 0.2) +
  ggtitle("pickup hour vs percentage") +
  facet_wrap("distance") +
  labs(x="pickup hour", y="percentage")
  