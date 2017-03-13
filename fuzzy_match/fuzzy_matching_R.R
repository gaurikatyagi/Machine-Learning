file_1 <- read.csv("sample1.csv", stringsAsFactors = FALSE)
file_2 <- read.csv("sample2.csv", stringsAsFactors = FALSE)

# Creates a matrix with the Standard Levenshtein distance between the name fields of both sources
distance_bw_names <- adist(file_1$name,file_2$Person.Name, partial = TRUE, ignore.case = TRUE)

#taking pairs with minimum distance, finding row wise
minimum_distance <- apply(distance_bw_names, 1, min)

matches <- NULL
for (i in 1:nrow(distance_bw_names)){
  # column index for minimum in each row of minimum_distance
  # find the ith element of minimum_distance  in the ith row of distances matrix
  file_2_i <- match(minimum_distance[i], distance_bw_names[i, ]) 
  print(file_2_i)
}
