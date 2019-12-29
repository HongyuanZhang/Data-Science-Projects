#336 Project 
#Hongyuan Zhang and Ziwen Chen

#load dataset
library(readxl)
GSP = read_excel("~/336 Project/GSP_Data_set.xlsx")
GSP_CRS = read_excel("~/336 Project/GSP_Data_set.xlsx",sheet = "CRS_INFO")

#merge datasets
total_ds=merge(GSP, GSP_CRS)

#check number of entries for each subject
#CHEM is one of the most popular subject
nrow(total_ds[total_ds$STC_SUBJECT=="BIO",]) #3205
nrow(total_ds[total_ds$STC_SUBJECT=="CHM",]) #2886
nrow(total_ds[total_ds$STC_SUBJECT=="CSC",]) #1030
nrow(total_ds[total_ds$STC_SUBJECT=="MAT",]) #4756
nrow(total_ds[total_ds$STC_SUBJECT=="PHY",]) #1284
nrow(total_ds[total_ds$STC_SUBJECT=="PSY",]) #1897

#check BIO/CHM 100-level and 200-level observation distributions
#CHEM has a more even 100/200-level distribution
nrow(total_ds[total_ds$CRS_NO=='129',]) #1693
nrow(total_ds[total_ds$CRS_NO=='221',]) #1193
nrow(total_ds[total_ds$CRS_NO=='150',]) #2224
nrow(total_ds[total_ds$CRS_NO=='251',]) #981

#filter to get only chem classes
chem_ds=total_ds[total_ds$STC_SUBJECT=="CHM" & !is.na(total_ds$STC_GRD_PTS),]

#filter out transferred grades
chem_ds=chem_ds[chem_ds$STC_CRED_TYPE=='G',]

#filter out audit, satisfactory and withdrawls passing
chem_ds=chem_ds[!chem_ds$VER_GRD=='AU' & !chem_ds$VER_GRD=='S' & !chem_ds$VER_GRD=='WP',]

#convert sat to act
sat2act = function(sat) {
  if (is.na(sat)) {
    return(NA)
  }
  sats=c(1600,1590,1530,1480,1430,1390,1350,1320,1280,1240,1200,1160,1120,1080,1040,1010,970,930,890,850,810,760,710,660,610,550,0)
  acts=seq(36,11,-1)
  for (i in seq(1,26)) {
    if (sat<=sats[i] & sat>sats[i+1]) {
      return(acts[i])
    }
  }
}

satm2actm = function(satm) {
  if (is.na(satm)) {
    return(NA)
  }
  sats=c(800,790,780,760,730,700,680,660,640,620,600,580,560,540,520,500,480,460,440,410,390,360,330,300,280,260,0)
  acts=seq(36,11,-1)
  for (i in seq(1,26)) {
    if (satm<=sats[i] & satm>sats[i+1]) {
      return(acts[i])
    }
  }
}

act_converted = sapply(chem_ds$`APP SAT TOT`,FUN=sat2act)
actm_converted = sapply(chem_ds$`APP SATM EVAL`,FUN=satm2actm)

combine_acts = function(act,converted_act) {
  ret=c()
  for (i in seq(1,length(act))) {
    if (is.na(act[i])) {
      ret=c(ret,converted_act[i])
    } else {
      ret=c(ret,act[i])
    }
  }
  ret
}

combined_act=combine_acts(chem_ds$`APP ACT EVAL`,act_converted)
combined_actm=combine_acts(chem_ds$`App Act Math Eval`,actm_converted)
chem_ds=data.frame(chem_ds,combined_act,combined_actm)

# prepare gsp data
gsp=chem_ds$GSP
gsp=gsub("\\d+P","P",gsp)
gsp=gsub("\\d+I","I",gsp)
gsp[is.na(gsp)] = 'N'
chem_ds = data.frame(chem_ds, gsp)

#PrimaryAcadInterest
sci=c("MEDICINE","BIOLOGY","BIOLOGICAL CHEMISTRY","ENGINEERING", "GENERAL SCIENCE", "MATH", "CHEMISTRY", "PSYCHOLOGY", 
      "ENVIRONMENTAL SCIENCE", "PHYSICS", "COMPUTER SCIENCE","DENTISTRY",'VETERINARY MEDICINE')
non_sci=c("SPANISH", "ECONOMICS", "RUSSIAN", "THEATRE", "HISTORY","RELIGIOUS STUDIES", "EDUCATION", "POLITICAL SCIENCE",
          "INTERNATIONAL RELATIONS", "ENGLISH", "FRENCH", "LANGUAGE", "SOCIOLOGY", "MUSIC", "ANTHROPOLOGY", "CLASSICS",
          "BUSINESS", "PHILOSOPHY", "PRE-LAW", "WRITING", "ART", "AMERICAN STUDIES","CHINESE","GERMAN")

chem_ds=mutate(chem_ds,Interest_In_Science = ifelse(chem_ds$APP.Primary.Acad.Interest %in% sci, 1,
                                                    ifelse(chem_ds$APP.Primary.Acad.Interest %in% non_sci, 0, NA)))

#filter out obviously wrongly recorded entries
wrong_records=chem_ds[which((chem_ds$APP.High.School.Rank<1) |
                              (chem_ds$APP.High.School.Size<1) |
                              (chem_ds$APP.Decile.Percent==0) |
                              (chem_ds$APP.HSGPA==0)),]
chem_ds=anti_join(chem_ds, wrong_records)

#Drop races with too few observations (<10)
chem_ds=chem_ds[chem_ds$RACES=='A'|
                chem_ds$RACES=='AW'|
                chem_ds$RACES=='B'|
                chem_ds$RACES=='BW'|
                chem_ds$RACES=='H'|
                chem_ds$RACES=='HW'|
                chem_ds$RACES=='U'|
                chem_ds$RACES=='W',]

#Check if there is any variable with many NAs that is not so relevant
#615 missing for APP.HSGPA
#3 missing for APP.Total.ADM.Rating
#23 missing for combined_act and combined_actm
#219 missing for APP.HS.REGION 
#466 missing for APP.Decile.Percent
#(the same) 499 entries missing for APP.High.School.Rank and Size
#555 missing for X..to.4.year
#361 missing for Interest_In_Science
#Although these varaibles have many NAs, we chose not to drop any because we believe that they can be associated with grade points.

#fit full model (everything is in this model; we don't use this model because of too few degrees of freedom)
everything=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                APP.Total.ADM.Rating+combined_act+combined_actm+
                APP.HS.REGION+APP.Decile.Percent+
                APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
                APP.VISITED.CAMPUS+Interest_In_Science+DistHome+HSGPA+PARENT_STAT+
                FATHER_EDUC+MOTHER_EDUC+HS_TYPE+ACAD_ABILITY+DRIVE_ACHIEVE+
                MATH_ABILITY+INTELLECT_SELFCONF+SOCIAL_SELFCONF+
                SELF_UNDERSTANDING+PROB_MAJOR+CRS_NO, 
              data=chem_ds)

#Because CIRP survey data have too many NAs, if we include them we will lose many degrees of freedom. 
#Thus we do not consider any survey variable.
#Chose RACES instead of ETHNIC: more detailed and less NAs
model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+POSSE+APP.HSGPA+
           APP.Total.ADM.Rating+combined_act+combined_actm+
           APP.HS.REGION+APP.Decile.Percent+
           APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
           APP.VISITED.CAMPUS+Interest_In_Science+CRS_NO, 
           data=chem_ds)

#test significance of high school region=>p-val=0.6862=>drop it
region_dropped_subset=chem_ds[!is.na(chem_ds$APP.HS.REGION),]
full_model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+POSSE+APP.HSGPA+
                APP.Total.ADM.Rating+combined_act+combined_actm+
                APP.HS.REGION+APP.Decile.Percent+
                APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
                APP.VISITED.CAMPUS+Interest_In_Science+CRS_NO, 
                data=chem_ds)
region_dropped=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+POSSE+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_act+combined_actm+
                    APP.Decile.Percent+
                    APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
                    APP.VISITED.CAMPUS+Interest_In_Science+CRS_NO, 
                    data=region_dropped_subset)
anova(region_dropped,full_model)

#p-val for APP.VISITED.CAMPUSY=0.85771=>drop it
model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+POSSE+APP.HSGPA+
                APP.Total.ADM.Rating+combined_act+combined_actm+
                APP.Decile.Percent+
                APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
                APP.VISITED.CAMPUS+Interest_In_Science+CRS_NO, 
                data=chem_ds)

#Dropped POSSE, p-val in the following model is 0.61398 
model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+POSSE+APP.HSGPA+
           APP.Total.ADM.Rating+combined_act+combined_actm+
           APP.Decile.Percent+
           APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
           Interest_In_Science+CRS_NO, 
           data=chem_ds)

#p-val for High School Rank is 0.63763 in the following model, drop it
model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
           APP.Total.ADM.Rating+combined_act+combined_actm+
           APP.Decile.Percent+
           APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
           Interest_In_Science+CRS_NO, 
         data=chem_ds)

#test significance of 4 year %=>p-val=0.3771=>drop it
four_year_dropped_subset=chem_ds[!is.na(chem_ds$X..to.4.year),]
full_model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                      APP.Total.ADM.Rating+combined_act+combined_actm+
                      APP.Decile.Percent+
                      APP.High.School.Size+X..to.4.year+
                      Interest_In_Science+CRS_NO, 
                      data=chem_ds)
four_year_dropped=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                       APP.Total.ADM.Rating+combined_act+combined_actm+
                       APP.Decile.Percent+
                       APP.High.School.Size+
                       Interest_In_Science+CRS_NO, 
                     data=four_year_dropped_subset)
anova(four_year_dropped,full_model)

#Dropped High School Size, p-val in the following model is 0.989768
model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
           APP.Total.ADM.Rating+combined_act+combined_actm+
           APP.Decile.Percent+
           APP.High.School.Size+
           Interest_In_Science+CRS_NO, 
         data=chem_ds)

#Dropped combined_act, p-val in the following model is 0.337482
model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
           APP.Total.ADM.Rating+combined_act+combined_actm+
           APP.Decile.Percent+
           Interest_In_Science+CRS_NO, 
         data=chem_ds)

#test significance of gsp %=>p-val=0.5542=>drop it
full_model=lm(STC_GRD_PTS~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                APP.Total.ADM.Rating+combined_actm+
                APP.Decile.Percent+
                Interest_In_Science+CRS_NO, 
              data=chem_ds)
gsp_dropped=lm(STC_GRD_PTS~GEND+RACES+X1STGEN+APP.HSGPA+
                 APP.Total.ADM.Rating+combined_actm+
                 APP.Decile.Percent+
                 Interest_In_Science+CRS_NO, 
               data=chem_ds)
anova(gsp_dropped,full_model)

#test significance of RACES %=>p-val=0.006283=>DO NOT DROP!!!
races_dropped_subset=chem_ds[!is.na(chem_ds$RACES),]
full_model=lm(STC_GRD_PTS~GEND+RACES+X1STGEN+APP.HSGPA+
                APP.Total.ADM.Rating+combined_actm+
                APP.Decile.Percent+
                Interest_In_Science+CRS_NO, 
              data=chem_ds)
races_dropped=lm(STC_GRD_PTS~GEND+X1STGEN+APP.HSGPA+
                   APP.Total.ADM.Rating+combined_actm+
                   APP.Decile.Percent+
                   Interest_In_Science+CRS_NO, 
                 data=races_dropped_subset)
anova(races_dropped,full_model)

#final set of variables
model=lm(STC_GRD_PTS~GEND+RACES+X1STGEN+APP.HSGPA+
           APP.Total.ADM.Rating+combined_actm+
           APP.Decile.Percent+
           Interest_In_Science+CRS_NO, 
           data=chem_ds)
plot(model)

#left-skewed Normal Q-Q=>square the response variable
model=lm(STC_GRD_PTS^2~GEND+RACES+X1STGEN+APP.HSGPA+
           APP.Total.ADM.Rating+combined_actm+
           APP.Decile.Percent+
           Interest_In_Science+CRS_NO, 
           data=chem_ds)

#do a model comparison test to find p-value for RACES again=>p-val=0.01504
races_dropped_subset=chem_ds[!is.na(chem_ds$RACES),]
full_model=lm(STC_GRD_PTS^2~GEND+RACES+X1STGEN+APP.HSGPA+
                APP.Total.ADM.Rating+combined_actm+
                APP.Decile.Percent+
                Interest_In_Science+CRS_NO, 
              data=chem_ds)
races_dropped=lm(STC_GRD_PTS^2~GEND+X1STGEN+APP.HSGPA+
                   APP.Total.ADM.Rating+combined_actm+
                   APP.Decile.Percent+
                   Interest_In_Science+CRS_NO, 
                 data=races_dropped_subset)
anova(races_dropped,full_model)

#FINAL MODEL(No interactions)
model=lm(STC_GRD_PTS^2~GEND+RACES+X1STGEN+APP.HSGPA+
           APP.Total.ADM.Rating+combined_actm+
           APP.Decile.Percent+
           Interest_In_Science+CRS_NO, 
           data=chem_ds)
summary(model)
#check model assumptions
plot(model)
vif(model)
#plot CIs
library(sjPlot)
plot_model(model)

#ANOVA for categorical variables in the previous model and their interactions
interaction_subset=chem_ds[which(!is.na(chem_ds$GEND) &
                                   !is.na(chem_ds$RACES) & 
                                   !is.na(chem_ds$Interest_In_Science) &
                                   !is.na(chem_ds$X1STGEN) & 
                                   !is.na(chem_ds$CRS_NO)),]
model=lm(STC_GRD_PTS^2~GEND+RACES+X1STGEN+Interest_In_Science+CRS_NO+
           GEND:RACES+GEND:X1STGEN+GEND:Interest_In_Science+GEND:CRS_NO+
           RACES:X1STGEN+RACES:Interest_In_Science+RACES:CRS_NO+
           X1STGEN:Interest_In_Science+X1STGEN:CRS_NO+
           +Interest_In_Science:CRS_NO, 
           data=interatcion_subset)
anova(model)

nrow(interaction_subset) #check if this matches (total degrees of freedom of the model with interactions) + 1

#check model assumptions
plot(model) #all okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$GEND) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$RACES)#Not perfect, but given our limited data, accept it
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$Interest_In_Science) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$X1STGEN) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$CRS_NO) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$GEND:interaction_subset$RACES) #Race data is not perfect, accept it
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$GEND:interaction_subset$Interest_In_Science) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$GEND:interaction_subset$X1STGEN) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$GEND:interaction_subset$CRS_NO) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$RACES:interaction_subset$Interest_In_Science) #Race data is not perfect, accept it
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$RACES:interaction_subset$X1STGEN) #Race data is not perfect, accept it
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$RACES:interaction_subset$CRS_NO) #Race data is not perfect, accept it
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$Interest_In_Science:interaction_subset$X1STGEN) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$Interest_In_Science:interaction_subset$CRS_NO) #Okay
boxplot(interaction_subset$STC_GRD_PTS^2~interaction_subset$X1STGEN:interaction_subset$CRS_NO) #Okay

interaction.plot(interaction_subset$RACES,interaction_subset$GEND,interaction_subset$STC_GRD_PTS)
interaction.plot(interaction_subset$X1STGEN,interaction_subset$Interest_In_Science,interaction_subset$STC_GRD_PTS)
interaction.plot(interaction_subset$RACES,interaction_subset$X1STGEN,interaction_subset$STC_GRD_PTS)


#fit logistic regression for delayed risk factor
boxplot(chem_ds[chem_ds$CRS_NO=='129',]$STC_GRD_PTS) #find approximately 50% of data above and below 12
good_129=chem_ds[chem_ds$CRS_NO=='129' & chem_ds$STC_GRD_PTS>=12.00,] #get all good 129 grades
all_221=chem_ds[chem_ds$CRS_NO=='221' & chem_ds$PKID %in% good_129$PKID,] #get all 221 grades of the students who had a good 129 grade
boxplot(chem_ds[chem_ds$CRS_NO=='221',]$STC_GRD_PTS) #similar distribution to CHM 129, so use 12 again as the standard for goodness
all_221$GRD_GOOD=(all_221$STC_GRD_PTS >=12.00) #convert these 221 grades to binary variable (prepare for logistic regression)
table(all_221$RACES)
#Drop races with too few observations (<10)
all_221=all_221[all_221$RACES=='A'|
                  all_221$RACES=='B'|
                  all_221$RACES=='H'|
                  all_221$RACES=='U'|
                  all_221$RACES=='W',]

model_delay = glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_act+combined_actm+
                    APP.HS.REGION+APP.Decile.Percent+
                    APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
                    APP.VISITED.CAMPUS+Interest_In_Science, 
                    data=all_221, family=binomial(link="logit"))

#Test significance of 4 year%, p-val=0.9214417 (drop!)
four_year_dropped_subset=all_221[!is.na(all_221$X..to.4.year),]
full_model = glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                   APP.Total.ADM.Rating+combined_act+combined_actm+
                   APP.HS.REGION+APP.Decile.Percent+
                   APP.High.School.Rank+APP.High.School.Size+X..to.4.year+
                   APP.VISITED.CAMPUS+Interest_In_Science, 
                 data=all_221, family=binomial(link="logit"))
four_year_dropped=glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                        APP.Total.ADM.Rating+combined_act+combined_actm+
                        APP.HS.REGION+APP.Decile.Percent+
                        APP.High.School.Rank+APP.High.School.Size+
                        APP.VISITED.CAMPUS+Interest_In_Science, 
                      data=four_year_dropped_subset, family=binomial(link="logit"))
anova(four_year_dropped,full_model)
1-pchisq(0.9216, 4)

#Test significance of APP.HS.REGION, p-val=0.2031719 (drop!)
region_dropped_subset=all_221[!is.na(all_221$APP.HS.REGION),]
full_model = glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                   APP.Total.ADM.Rating+combined_act+combined_actm+
                   APP.HS.REGION+APP.Decile.Percent+
                   APP.High.School.Rank+APP.High.School.Size+
                   APP.VISITED.CAMPUS+Interest_In_Science, 
                 data=all_221, family=binomial(link="logit"))
region_dropped=glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                     APP.Total.ADM.Rating+combined_act+combined_actm+
                     APP.Decile.Percent+
                     APP.High.School.Rank+APP.High.School.Size+
                     APP.VISITED.CAMPUS+Interest_In_Science, 
                   data=region_dropped_subset, family=binomial(link="logit"))
anova(region_dropped,full_model)
1-pchisq(5.9466, 4)

#The p-val for APP.Decile.Percent is 0.9682 in the following model, drop it
model_delay = glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_act+combined_actm+
                    APP.Decile.Percent+
                    APP.High.School.Rank+APP.High.School.Size+
                    APP.VISITED.CAMPUS+Interest_In_Science, 
                  data=all_221, family=binomial(link="logit"))

#The p-val for combined_act is 0.8189 in the following model, drop it
model_delay = glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_act+combined_actm+
                    APP.High.School.Rank+APP.High.School.Size+
                    APP.VISITED.CAMPUS+Interest_In_Science, 
                  data=all_221, family=binomial(link="logit"))

#The Wald's p-val for Interest_In_Science is 0.73027 (drop!)
model_delay=glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                 APP.Total.ADM.Rating+combined_actm+
                 APP.High.School.Rank+APP.High.School.Size+
                 APP.VISITED.CAMPUS+Interest_In_Science, 
               data=all_221, family=binomial(link="logit"))

#The Wald's p-val for 1stGen is 0.89966 (drop!)
model_delay=glm(GRD_GOOD~GEND+RACES+gsp+X1STGEN+APP.HSGPA+
                 APP.Total.ADM.Rating+combined_actm+
                 APP.High.School.Rank+APP.High.School.Size+
                 APP.VISITED.CAMPUS, 
               data=all_221, family=binomial(link="logit"))

#The Wald's p-val for APP.VISITED.CAMPUS is 0.73356 (drop!)
model_delay=glm(GRD_GOOD~GEND+RACES+gsp+APP.HSGPA+
                 APP.Total.ADM.Rating+combined_actm+
                 APP.High.School.Rank+APP.High.School.Size+
                 APP.VISITED.CAMPUS, 
               data=all_221, family=binomial(link="logit"))

#Test significance of gsp, LRT p-val=0.4229294 (drop!)
full_model=glm(GRD_GOOD~GEND+RACES+gsp+APP.HSGPA+
                 APP.Total.ADM.Rating+combined_actm+
                 APP.High.School.Rank+APP.High.School.Size, 
               data=all_221, family=binomial(link="logit"))
gsp_dropped=glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                  APP.Total.ADM.Rating+combined_actm+
                  APP.High.School.Rank+APP.High.School.Size, 
                data=all_221, family=binomial(link="logit"))
anova(gsp_dropped,full_model)
1-pchisq(1.7211, 2)

#The Wald's p-val for High School Rank is 0.14753, drop it
model_delay = glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_actm+
                    APP.High.School.Rank+APP.High.School.Size, 
                  data=all_221, family=binomial(link="logit"))

#The Wald's p-val for High School Size is 0.381026, drop it
model_delay = glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_actm+
                    APP.High.School.Size, 
                  data=all_221, family=binomial(link="logit"))

#The Wald's p-val for ACT Math Score is 0.352068, drop it
model_delay = glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                    APP.Total.ADM.Rating+combined_actm, 
                  data=all_221, family=binomial(link="logit"))

#Should we drop GEND?
model_delay = glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                    APP.Total.ADM.Rating, 
                  data=all_221, family=binomial(link="logit")) #model before drop GEND
hoslem.test(model_delay$y, model_delay$fitted.values)
model_delay = glm(GRD_GOOD~RACES+APP.HSGPA+
                    APP.Total.ADM.Rating, 
                  data=all_221, family=binomial(link="logit")) #model with GEND dropped
hoslem.test(model_delay$y, model_delay$fitted.values) #p-value decreased
#So DO NOT drop GEND

#Test significance of RACES, LRT p-val=0.009011305 (DO NOT drop!)
races_dropped_subset=all_221[!is.na(all_221$RACES),]
full_model=glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                 APP.Total.ADM.Rating, 
               data=all_221, family=binomial(link="logit"))
races_dropped=glm(GRD_GOOD~GEND+APP.HSGPA+
                    APP.Total.ADM.Rating, 
                  data=races_dropped_subset, family=binomial(link="logit"))
anova(races_dropped,full_model)
1-pchisq(13.516, 4)

#final model; variables in this model may be real delayed risk factors
model_delay = glm(GRD_GOOD~GEND+RACES+APP.HSGPA+
                    APP.Total.ADM.Rating, 
                  data=all_221, family=binomial(link="logit"))
summary(model_delay)
vif(model_delay) 
hoslem.test(model_delay$y, model_delay$fitted.values) #p-value=0.8895
plot_model(model_delay)




