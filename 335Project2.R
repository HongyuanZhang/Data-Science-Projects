#MAT-335 Project 2 R script
#Ziwen Chen and Hongyuan Zhang

#required for laplace functions
library(rmutil)
#required for calculating kurtosis
library(moments)

##################1#############################
# plot standard normal distribution and standard laplace distribution
x <- seq(-3, 3, by = .01)
y <- dnorm(x)
z <- dlaplace(x)
plot(x, y, type="l", lwd=1, ylim=range(0, 0.5), ylab="pdf", main = "Standard Normal and Standard Laplace")
lines(x, z, type='l', lwd=1, col="blue")

#given a sample size n, calculate the probability that a laplace sample's kurtosis is smaller than or equal to a normal sample's kurtosis
lsmallerthann<-function(size){
  count=0
  # repeat 10000 times to approximate probability
  for(i in c(1:10000)) {
    ldata<-rlaplace(size)
    l_k<-kurtosis(ldata)
    ndata<-rnorm(size)
    n_k<-kurtosis(ndata)
    if(l_k<=n_k){
      count=count+1
    }
  }
  return (count/10000)
}

#approximate cutoff value, given a collection of 10000 normal kurtosis and a collection of 10000 laplace kurtosis
approx_cutoff<-function(n_kurs, l_kurs){
  #we know that the cutoff value is approximately between 2.85 and 4
  for(i in seq(from=2.85, to=4, by=0.001)){
    count1=0
    count2=0
    for(j in c(1:10000)){
      if(n_kurs[j]>=i){
        count1=count1+1
      }
      if(l_kurs[j]<=i){
        count2=count2+1
      }
    }
    #if the counts differ by less than 1, we've found our cutoff value, i
    #otherwise, continue increasing i until we find a value that satisfies abs(count1-count2)<=1
    #in the case of n=50, consider changing the condition to abs(count1-count2)<=3 or decreasing the increment of i to 0.0001 
    if(abs(count1-count2)<=1){
      # print error rate
      print(count1/10000)
      # print cutoff value
      return(i)
      }
  }
  return(0)
}

#plot the sampling distribution of kurtosis for both distributions, given a sample size
plot_kurtosis<-function(sample_size){
  #generate 10000 normal samples of size sample_size, store their kutosis in n_kur
  n_kur<-c()
  for(i in c(1:10000)){
    n_data<-rnorm(sample_size)
    n_kur<-c(n_kur, kurtosis(n_data))
  }
  #estimate density and plot
  n_d <- density(n_kur)
  plot(n_d, xlim=range(2:7))
  #generate 10000 laplace samples of size sample_size, store their kutosis in l_kur
  l_kur<-c()
  for(i in c(1:10000)){
    l_data<-rlaplace(sample_size)
    l_kur<-c(l_kur, kurtosis(l_data))
  }
  #estimate density and plot
  l_d <- density(l_kur)
  lines(l_d, col="blue")
  #print and plot cutoff value
  if((cutoff=approx_cutoff(n_kur, l_kur)) != 0){
    print(cutoff)
    abline(v=cutoff, col="green")
  }
}

#verify that probability that the kurtosis of a Laplace sample of size 50 is below 2.85 is around 5%
#and that probability that the kurtosis of a Normal sample of size 50 is above 4 is around 5%
error_rate_for_not_in_between<-function(sample_size){
  #generate 10000 normal samples of size sample_size, store their kutosis in n_kur
  n_kur<-c()
  for(i in c(1:10000)){
    n_data<-rnorm(sample_size)
    n_kur<-c(n_kur, kurtosis(n_data))
  }
  #generate 10000 laplace samples of size sample_size, store their kutosis in l_kur
  l_kur<-c()
  for(i in c(1:10000)){
    l_data<-rlaplace(sample_size)
    l_kur<-c(l_kur, kurtosis(l_data))
  }
  #count the times n_kur>4 and l_cur<2.85
  count1=0
  count2=0
  for(i in c(1:10000)){
    if(n_kur[i]>4){count1=count1+1}
    if(l_kur[i]<2.85){count2=count2+1}
  }
  #print probabilities
  print(count1/10000)
  print(count2/10000)
}

##################2#############################

#define a function that generates a residual standard deviation for a sample
generateSigma = function(isNormal) {
  #number of data points in one sample
  n=4000
  #bins for data value (middle value of each bin)
  brks = seq(-6,5.9,0.1)+0.05
  
  #generate data from standard normal or laplace
  data = c()
  if (isNormal) {
    data=rnorm(n)
  } else {
    data=rlaplace(n)
  }
  
  #calculate frequency for each bin
  freqs=c()
  for (b in brks) {
    count=0
    for (d in data) {
      if ((b-0.05<=d) & (d<b+0.05)) {
        count = count + 1
      }
    }
    freqs=c(freqs,count/n)
  }
  
  #plot frequency on a log scale
  #plot(brks,freqs,main="Sample Normal Distribution",xlab="",ylab="Proportion",pch=20)
  #plot(brks,log(freqs),main="Sample Normal Distribution (Log scale)",xlab="",ylab="Proportion",pch=20)
  
  #require(reshape2)
  #take log of the frequencies and revert the right half
  logfreqs=log(freqs)
  logfreqs[which(logfreqs==-Inf)] = NA
  diff=max(logfreqs,na.rm=T)-min(logfreqs,na.rm=T)
  logfreqs[((length(brks)/2)+1):(length(brks))]=
    2*min(logfreqs,na.rm=T)+2*diff-logfreqs[((length(brks)/2)+1):(length(brks))]
  
  #combine frequencies and value bins into a data frame
  df = data.frame(brks,logfreqs)
  colnames(df) <- c("value", "frequency")
  #View(df)
  #plot(df,main="Sample Normal Distribution (Log scale, flipped)",xlab="",ylab="Proportion",pch=20)
  
  #apply linear model and store the result
  lmv=lm(frequency~value,df,na.action=na.omit)
  #a=summary(lmv)$coefficients[1,1]
  #b=summary(lmv)$coefficients[2,1]
  #abline(a=a,b=b,col="red",lwd=2)
  rsigma=sigma(lmv)
  
  return(rsigma)
}

#collect normal and laplace sigmas using 1000 samples each
normalsigmas = c()
laplacesigmas = c()
for (i in 1:1000) {
  n=generateSigma(TRUE)
  normalsigmas=c(normalsigmas,n)
  l=generateSigma(FALSE)
  laplacesigmas=c(laplacesigmas,l)
}

#approximate error rate and cutoff value, similar to approx_cutoff in Part 1
#from histograms, we know that the cutoff value is approximately between 0.45 and 0.55
for(i in seq(from=0.45, to=0.55, by=0.0001)){
  count1=0
  count2=0
  for(j in c(1:1000)){
    if(normalsigmas[j]<=i){
      count1=count1+1
    }
    if(laplacesigmas[j]>=i){
      count2=count2+1
    }
  }
  #if the counts differ by less than 1, we've found our cutoff value, i
  #otherwise, continue increasing i until we find a value that satisfies abs(count1-count2)<=1
  if(abs(count1-count2)<=1){
    # print error rate
    print(count1/1000)
    # print cutoff value
    print(i)
    break
  }
}

#draw the sigma sampling distribution
hist(normalsigmas, main="Histogram of Residual Standard Deviations",
     xlim=c(0.2,0.8),col=rgb(0,0,1,1/4), breaks = seq(0.2,0.8,0.01),
     xlab = "Residual Standard Deviations")
hist(laplacesigmas, xlim=c(0.2,0.8), add=T,col=rgb(1,0,0,1/4),
     breaks = seq(0.2,0.8,0.01))
legend("topleft", inset=.02, c("Normal","Laplace"), fill=c(rgb(0,0,1,1/4),rgb(1,0,0,1/4)), horiz=F, cex=0.8)

