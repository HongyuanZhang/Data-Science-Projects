import pandas as pd
import numpy as np
import kmapper
import sklearn
from kmapper import KeplerMapper
import re
import scipy as sp
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LogisticRegression

'''Keep the desirable features and drop data points with missing value(s) '''
df = pd.read_csv('MetObjects.csv', encoding="ISO-8859-1")
feature_names = ['Is Highlight', 'Is Public Domain', 'Object ID', 'Department', 'Object Name', 'Artist Begin Date','Artist End Date', 'Object Begin Date', 'Object End Date', 'Medium', 'Dimensions', 'Credit Line', 'Classification']
X = df[feature_names].dropna(axis=0)
''' Filter out data points whose date information doesn't make sense '''
sub=X[X['Artist Begin Date'] != X['Artist End Date']]
sub=sub[sub['Object Begin Date'] < sub['Object End Date']]
'''Focus on artworks with 2 measurements.
   Extract numbers from strings. Replace Dimensions with area (product of the two extracted numbers).
   Record Length and Width for each artwork '''
length=[]
width=[]
for index, row in sub.iterrows():
    l1 = re.findall('\((.*?) cm', row[10])
    if len(l1) == 1:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", l1[0])
        if len(nums) == 2:
            sub.loc[index, 'Dimensions'] = float(nums[0]) * float(nums[1])
            length.append(float(nums[0]))
            width.append(float(nums[1]))
        else:
            sub.drop(index, inplace=True)
    else:
        sub.drop(index, inplace=True)
sub['Length']=length
sub['Width']=width
'''Extract Artist Begin Date and Artist End Date. 
   If there is more than one number in this field, take the average. '''
delimiters = ' ', '|'
regexPattern = '|'.join(map(re.escape, delimiters))
for index, row in sub.iterrows():
    l2 = list(filter(lambda a: a != '', re.split(regexPattern, row[5])))
    for x in l2:
        if '-' in x and x[0] != '-':
            l2[l2.index(x)] = int(str.split(x, '-')[0])
        elif '/' in x:
            l2[l2.index(x)] = int(str.split(x, '/')[2])
        else:
            l2[l2.index(x)] = int(x)
    sub.loc[index, 'Artist Begin Date'] = int(np.mean(l2))

for index, row in sub.iterrows():
    l3 = list(filter(lambda a: a != '', re.split(regexPattern, row[6])))
    for x in l3:
        if '-' in x and x[0] != '-':
            l3[l3.index(x)] = int(str.split(x, '-')[0])
        elif '/' in x:
            l3[l3.index(x)] = int(str.split(x, '/')[2])
        elif x == '9999':
            l3[l3.index(x)] = 1999
        else:
            l3[l3.index(x)] = int(x)
    sub.loc[index, 'Artist End Date'] = int(np.mean(l3))

sub.to_csv('fullset.csv')

''' To be deleted. '''
df = pd.read_csv('fullset.csv')
sub = df.drop(df.columns[0], axis=1)
sub = sub.drop(columns=['Is Highlight'])

''' Initialize dictionaries for 'Object Name', 'Medium', 'Credit Line', 'Classification' '''
mydict_name={}
mydict_medium={}
mydict_cl={}
mydict_class={}

''' Convert the type of these variables (plus 'Department') to 'category', so that we can easily assign them numerical values '''
sub['Department'] = sub['Department'].astype('category')
sub['Object Name'] = sub['Object Name'].astype('category')
sub['Medium'] = sub['Medium'].astype('category')
sub['Credit Line'] = sub['Credit Line'].astype('category')
sub['Classification'] = sub['Classification'].astype('category')

''' Create new columns in sub and set them to the numerical values of the corresponding columns '''
sub['depart_cat'] = sub['Department'].cat.codes
sub['name_cat'] = sub['Object Name'].cat.codes
sub['med_cat'] = sub['Medium'].cat.codes
sub['cl_cat'] = sub['Credit Line'].cat.codes
sub['class_cat'] = sub['Classification'].cat.codes

''' Record the correspondence between the numerical values and strings for each of the variables.
    We don't do so for 'Deparment' because of the way we give a subscore for it (merely comparing if two artworks have the same value). '''
delimiters = ' and ', ' or ', ' ', ',', ';', '&', '(?)', '(', ')', '/', '|', '.'
regexPattern = '|'.join(map(re.escape, delimiters))
for idx, item in enumerate(sub['name_cat']):
    mydict_name[item]=list(filter(lambda a: a != '', re.split(regexPattern, sub['Object Name'][idx])))
for idx, item in enumerate(sub['med_cat']):
    mydict_medium[item]=list(filter(lambda a: a != '', re.split(regexPattern, sub['Medium'][idx])))
for idx, item in enumerate(sub['cl_cat']):
    delims = ', ', '; '
    rePattern = '|'.join(map(re.escape, delims))
    mydict_cl[item]=list(filter(lambda a: a != '', re.split(rePattern, sub['Credit Line'][idx])))
for idx, item in enumerate(sub['class_cat']):
    mydict_class[item]=list(filter(lambda a: a != '', re.split(regexPattern, sub['Classification'][idx])))
    '''For Classification, we need to further refine the lists of meaningful words, 
        since the words after '-' seem similar to an artwork's Object Name and we decided to discard the part after '-'''
    mydict_class[item]=list(map(lambda x: x.split('-')[0], mydict_class.get(item)))
'''
selected=['Is Public Domain', 'Object Begin Date', 'Object End Date', 'Dimensions','Length', 'Width', 'Artist Begin Date', 'Artist End Date']
sub[selected].to_csv('logistic.csv')
'''
''' Drop the columns with string values and keep only the numerical columns '''
sub = sub.drop(columns=['Department', 'Object Name', 'Medium', 'Credit Line', 'Classification'])
'''Calculate the maximum differences, which will be used in mydist to limit subscores between 0-1.'''
max_begin=max(sub['Object Begin Date'])-min(sub['Object Begin Date'])
max_end=max(sub['Object End Date'])-min(sub['Object End Date'])
max_begin_artist=max(sub['Artist Begin Date'])-min(sub['Artist Begin Date'])
max_end_artist=max(sub['Artist End Date'])-min(sub['Artist End Date'])
max_dim=max(sub['Dimensions'])-min(sub['Dimensions'])
max_len=max(sub['Length'])-min(sub['Length'])
max_wid=max(sub['Width'])-min(sub['Width'])

''' Jaccard index. 
    a: a set
    b: a set
    Output: Jaccard index for set a and set b.
    '''
def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

''' Determine similarity between two artworks in terms of some categorical feature
    x: (list of strings) some feature of an artwork
    y: (list of strings) the same feature of another artwork
    v: (string) this feature's name
    Output: a score denoting similarity (1: remote 0: close)
    This function computes the Jaccard distance for x and y.
'''
def cat_dist(x, y, v):
    return 1-jaccard(x, y)

'''
Calculate the distance between two artworks.
x: a numpy.ndarray/list that contains an artwork's 13 features
y: a numpy.ndarray/list that contains an artwork's 13 features
Output: distance between 0-13 (0: close 13: remote)
'''
def mydist(x, y):
    sc1=0
    sc2=0
    sc3=0
    sc4=0
    sc5=0
    sc6=0
    sc7=0
    sc8=0
    sc9=0
    sc10=0
    sc11=0
    sc12=0
    sc13=0
    ''' Check if the two artworks are the same based on Object Id '''
    if x[0] != y[0]:
        sc1 = 1
    ''' Calculate a score for 'Artist Begin Date' '''
    sc2=abs(x[1]-y[1])/max_begin_artist
    ''' Calculate a score for 'Artist End Date' '''
    sc3=abs(x[2]-y[2])/max_end_artist
    ''' Calculate a score for 'Object Begin Date' '''
    sc4=abs(x[3]-y[3])/max_begin
    ''' Calculate a score for 'Object End Date' '''
    sc5=abs(x[4]-y[4])/max_end
    ''' Calculate a score for 'Dimensions' '''
    sc6 = abs(x[5]-y[5])/max_dim
    sc7 = abs(x[6]-y[6])/max_len
    sc8 = abs(x[7]-y[7])/max_wid
    ''' Calculate a score for 'Deparment', since there are only 19 different values for this feature,
    we simply check if two artworks have the same value'''
    if x[8] != y[8]:
        sc9=1
    ''' Calculate a score for 'Object Name' using cat_dist '''
    sc10=cat_dist(set(mydict_name.get(x[9])), set(mydict_name.get(y[9])), 'Object Name')
    ''' Calculate a score for 'Medium' using cat_dist '''
    sc11= cat_dist(set(mydict_medium.get(x[10])), set(mydict_medium.get(y[10])), 'Medium')
    ''' Calculate a score for 'Credit Line' using cat_dist '''
    sc12 = cat_dist(set(mydict_cl.get(x[11])), set(mydict_cl.get(x[11])), 'Credit Line')
    ''' Calculate a score for 'Classification' using cat_dist '''
    sc13 = cat_dist(set(mydict_class.get(x[12])), set(mydict_class.get(y[12])), 'Classification')
    '''We add up the scores to give the distance between x and y.
    print([sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12)'''
    return sc1+sc2+sc3+sc4+sc5+sc6+sc7+sc8+sc9+sc10+sc11+sc12+sc13

'''lenses: eccentricity, L-infinity centrality'''
def eccentricity(x):
    dist_sum=0
    for y in sub:
        dist_sum=dist_sum+mydist(x,y)
    return dist_sum/len(sub)

def l_infinity(x):
    max_dist=0
    for y in sub:
        if mydist(x, y) > max_dist:
            max_dist=mydist(x, y)
    return max_dist

''' Focus on a random subset of 3000. 
    Define the color function for Mapper output graph as Is Public Domain, the variable which we are interested in predicting. 
    Then drop it from the sample. '''
subset=sub.sample(3000, random_state=999)
my_colors=np.array(subset['Is Public Domain'])
subset=np.array(subset.drop(columns=['Is Public Domain']))

'''Initialize Mapper'''
mapper: KeplerMapper = kmapper.KeplerMapper(verbose=2)
'''Initialize MDS, our lens function for Mapper.
   Since we are using mydist, a custom metric, we set dissmilarity. '''
mds=sklearn.manifold.MDS(dissimilarity='precomputed', random_state=88)
'''Using mydist, compute a distance matrix for our sample and save it.'''
dist_matrix = sp.spatial.distance.squareform(sp.spatial.distance.pdist(subset, metric=mydist))
np.savetxt('3000pcs.csv', dist_matrix, delimiter=',')
'''Apply MDS to our sample to get our lens.'''
lens=mds.fit_transform(dist_matrix)
'''Using DBSCAN and our metric, generate simplicial complex.'''
simplicial_complex= mapper.map(lens,
                                subset,
                                nr_cubes=10,
                                overlap_perc=0.4,
                                clusterer=sklearn.cluster.DBSCAN(eps=5, metric=mydist, algorithm='brute', min_samples=10))
''' Output the simplicial complex, colored by 'Is Public Domain', as a webpage. '''
html = mapper.visualize(simplicial_complex, path_html="art.html", color_function=my_colors)

''' Examples:
1. d(x, y) >= 0
for x in subset:
    for y in subset:
        if mydist(x, y) < 0:
            print(x, y)
            
2. d(x, x) = 0
for x in subset:
    if mydist(x, x) != 0:
        print(x)

3. d(x, y) = 0 => x = y
This is clear since no artworks share the same Object ID, and the way we calculate distance says that if x, y do not have the same Object ID, we assign a subscore of 1.

4. d(x, y) = d(y, x)
for x in subset:
    for y in subset:
        if mydist(x, y) != mydist(y, x):
            print(x, y)

5. d(x, y) + d(y, z) >= d(x, z)
def triangle_inequality(x, y, z):
    if mydist(x, y) + mydist(y, z) >= mydist(x, z):
        return True
    else:
        print(x, y, z)
        return False
for x in subset:
    for y in subset:
        for z in subset:
            triangle_inequality(x, y, z)
            
The following codes are for determining the subset of the dataset to work on.
df2=df[feature_names].dropna(axis=0)
num=num2=0
for str in df2['Dimensions']:
    if len(re.findall('\((.*?) cm', str)) == 1:
        num = num + 1
    else:
        num2 = num2 + 1

num=num2=num3=num4=0
for str in df2['Dimensions']:
    l = re.findall('\((.*?) cm', str)
    if len(l) == 1:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", l[0])
        if len(nums) == 1:
            num = num + 1
        elif len(nums) == 2:
            num2 = num2 + 1
        elif len(nums) == 3:
            num3 = num3 + 1
        else:
            num4 = num4 + 1
'''
red_nodes=[]
red_nodes.append('cube20_cluster0')
red_nodes.append('cube31_cluster0')
red_nodes.append('cube21_cluster0')
red_nodes.append('cube20_cluster1')
red_nodes.append('cube30_cluster0')
red_nodes.append('cube31_cluster1')
red_nodes.append('cube41_cluster0')
red_nodes.append('cube40_cluster0')
red_nodes.append('cube51_cluster0')
red_nodes.append('cube50_cluster0')
red_nodes.append('cube61_cluster0')
blue_nodes=[]
blue_nodes.append('cube67_cluster0')
blue_nodes.append('cube57_cluster0')
blue_nodes.append('cube58_cluster0')
blue_nodes.append('cube48_cluster0')
blue_nodes.append('cube38_cluster0')
blue_nodes.append('cube47_cluster0')
blue_nodes.append('cube37_cluster0')
blue_nodes.append('cube27_cluster0')
blue_nodes.append('cube36_cluster0')
blue_nodes.append('cube26_cluster0')
blue_nodes.append('cube25_cluster0')
blue_nodes.append('cube16_cluster0')
blue_nodes.append('cube15_cluster0')
red_index=[]
for x in red_nodes:
    red_index.extend(simplicial_complex.get('nodes').get(x))
red_index=list(set(red_index))
red_pd=[]
for x in red_index:
    red_pd.append(my_colors[x])
len(red_pd)
#276
sum(red_pd)
#225
blue_index = []
for x in blue_nodes:
    blue_index.extend(simplicial_complex.get('nodes').get(x))
blue_index = list(set(blue_index))
blue_pd = []
for x in blue_index:
    blue_pd.append(my_colors[x])
len(blue_pd)
#1049
sum(blue_pd)
#275
count = np.array([225, 275])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#0.000
abd_red=[]
for x in red_index:
    abd_red.extend(list(sub[sub['Object ID']==subset[x][0]]['Artist Begin Date']))
abd_blue=[]
for x in blue_index:
    abd_blue.extend(list(sub[sub['Object ID']==subset[x][0]]['Artist Begin Date']))
sp.stats.ranksums(abd_red, abd_blue)
#RanksumsResult(statistic=-11.606060878281362, pvalue=3.8389682437780603e-31) Significant
aed_red=[]
for x in red_index:
    aed_red.extend(list(sub[sub['Object ID']==subset[x][0]]['Artist End Date']))
aed_blue=[]
for x in blue_index:
    aed_blue.extend(list(sub[sub['Object ID']==subset[x][0]]['Artist End Date']))
sp.stats.ranksums(aed_red, aed_blue)
#RanksumsResult(statistic=-10.265227229606868, pvalue=1.0108271873662676e-24) Significant
np.median(aed_red)
#1807.5
np.median(aed_blue)
#1920.0
np.median(abd_red)
#1744.0
np.median(abd_blue)
#1850.0
obd_red=[]
for x in red_index:
    obd_red.extend(list(sub[sub['Object ID'] == subset[x][0]]['Object Begin Date']))
obd_blue = []
for x in blue_index:
    obd_blue.extend(list(sub[sub['Object ID'] == subset[x][0]]['Object Begin Date']))
sp.stats.ranksums(obd_red, obd_blue)
#RanksumsResult(statistic=-11.489551203376761, pvalue=1.4888338343397409e-30) Significant
oed_red=[]
for x in red_index:
    oed_red.extend(list(sub[sub['Object ID'] == subset[x][0]]['Object End Date']))
oed_blue = []
for x in blue_index:
    oed_blue.extend(list(sub[sub['Object ID'] == subset[x][0]]['Object End Date']))
sp.stats.ranksums(oed_red, oed_blue)
#RanksumsResult(statistic=-10.972948365635876, pvalue=5.1563080839906435e-28) Significant
np.median(obd_red)
#1760.0
np.median(obd_blue)
#1880.0
np.median(oed_red)
#1792.5
np.median(oed_blue)
#1890.0
dim_red=[]
for x in red_index:
    dim_red.extend(list(sub[sub['Object ID'] == subset[x][0]]['Dimensions']))
dim_blue = []
for x in blue_index:
    dim_blue.extend(list(sub[sub['Object ID'] == subset[x][0]]['Dimensions']))
sp.stats.ranksums(dim_red, dim_blue)
#RanksumsResult(statistic=16.159308476905167, pvalue=9.76509242707288e-59) Significant
len_red=[]
for x in red_index:
    len_red.extend(list(sub[sub['Object ID'] == subset[x][0]]['Length']))
len_blue = []
for x in blue_index:
    len_blue.extend(list(sub[sub['Object ID'] == subset[x][0]]['Length']))
sp.stats.ranksums(len_red, len_blue)
#RanksumsResult(statistic=14.8747848911398, pvalue=4.805270833716341e-50)
wid_red=[]
for x in red_index:
    wid_red.extend(list(sub[sub['Object ID'] == subset[x][0]]['Width']))
wid_blue = []
for x in blue_index:
    wid_blue.extend(list(sub[sub['Object ID'] == subset[x][0]]['Width']))
sp.stats.ranksums(wid_red, wid_blue)
#RanksumsResult(statistic=16.558429252469864, pvalue=1.3918027191571795e-61)
np.median(dim_red)
#1893.7150000000001
np.median(dim_blue)
#109.06
np.median(len_red)
#41.599999999999994
np.median(len_blue)
#11.2
np.median(wid_red)
#42.7
np.median(wid_blue)
#8.9
pd.DataFrame(subset)[8].unique()
#array([ 4., 13.,  3., 10.,  0.,  6.,  5., 12.,  2.,  1., 14.,  8.,  9.])
red_0=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 0:
        red_0.append(1)
    else:
        red_0.append(0)
blue_0=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 0:
        blue_0.append(1)
    else:
        blue_0.append(0)
count = np.array([59, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#5.6476517832074895e-53
red_1=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 1:
        red_1.append(1)
    else:
        red_1.append(0)
blue_1=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 1:
        blue_1.append(1)
    else:
        blue_1.append(0)
count = np.array([2, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#0.006
red_2=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 2:
        red_2.append(1)
    else:
        red_2.append(0)
blue_2=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 2:
        blue_2.append(1)
    else:
        blue_2.append(0)
count = np.array([0, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#Invalid
red_3=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 3:
        red_3.append(1)
    else:
        red_3.append(0)
blue_3=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 3:
        blue_3.append(1)
    else:
        blue_3.append(0)
count = np.array([47, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#3.5429344587157824e-42
red_4=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 4:
        red_4.append(1)
    else:
        red_4.append(0)
blue_4=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 4:
        blue_4.append(1)
    else:
        blue_4.append(0)
count = np.array([0, 1048])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#8.578128248147642e-289
red_5=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 5:
        red_5.append(1)
    else:
        red_5.append(0)
blue_5=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 5:
        blue_5.append(1)
    else:
        blue_5.append(0)
count = np.array([82, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#3.157917716649403e-74
red_6=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 6:
        red_6.append(1)
    else:
        red_6.append(0)
blue_6=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 6:
        blue_6.append(1)
    else:
        blue_6.append(0)
count = np.array([69, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#4.018824778080052e-62
red_8=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 8:
        red_8.append(1)
    else:
        red_8.append(0)
blue_8=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 8:
        blue_8.append(1)
    else:
        blue_8.append(0)
#invalid
red_9=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 9:
        red_9.append(1)
    else:
        red_9.append(0)
blue_9=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 9:
        blue_9.append(1)
    else:
        blue_9.append(0)
#invalid
red_10=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 10:
        red_10.append(1)
    else:
        red_10.append(0)
blue_10=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 10:
        blue_10.append(1)
    else:
        blue_10.append(0)
count = np.array([7, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#2.3198662946472394e-07
red_12=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 12:
        red_12.append(1)
    else:
        red_12.append(0)
blue_12=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 12:
        blue_12.append(1)
    else:
        blue_12.append(0)
count = np.array([7, 1])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#3.1990577036414517e-06
red_13=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 13:
        red_13.append(1)
    else:
        red_13.append(0)
blue_13=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 13:
        blue_13.append(1)
    else:
        blue_13.append(0)
count = np.array([3, 0])
nobs = np.array([276, 1049])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))
#0.0007234362029969521
red_14=[]
for x in red_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 14:
        red_14.append(1)
    else:
        red_14.append(0)
blue_14=[]
for x in blue_index:
    if list(sub[sub['Object ID'] == subset[x][0]]['depart_cat'])[0] == 14:
        blue_14.append(1)
    else:
        blue_14.append(0)
#invalid
#reduced=['depart_4']
'''Compare full model with reduced model'''
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(pd.DataFrame(subset)[13], my_colors, test_size=0.2)
model = logreg.fit(np.reshape(np.array(X_train),(-1, 1)), y_train)
predictions = logreg.predict(np.reshape(np.array(X_test), (-1,1)))
model.score(np.reshape(np.array(X_test), (-1, 1)), y_test)
#0.5216666666666666
X=subset[['Artist Begin Date', 'Artist End Date', 'Object Begin Date', 'Object End Date', 'Dimensions', 'Length', 'Width', 'American Decorative Arts', 'Arms and Armor', 'Arts of Africa, Oceania, and the Americas', 'Asian Art', 'Drawings and Prints', 'European Paintings', 'European Sculpture and Decorative Arts', 'Islamic Art', 'Medieval Art', 'Modern and Contemporary Art', 'Photographs', 'Robert Lehman Collection']]
model.score(X_test, y_test)
#0.7383333333333333

#For sub, reduced model accuracy score is 0.5530340897730681, and full model accauray score is 0.7257822653204039

''' Artist Begin Date & Artist End Date v.s. N.A. '''
'''df = pd.read_csv('MetObjects.csv', encoding="ISO-8859-1")
feature_names = ['Is Public Domain', 'Object ID', 'Department', 'Object Name', 'Artist Begin Date', 'Artist End Date', 'Object Begin Date', 'Object End Date', 'Medium', 'Dimensions', 'Credit Line', 'Classification']
X = df[feature_names]
X=X[X['Object Begin Date'] < X['Object End Date']]
X_na=X[pd.isna(X['Artist Begin Date'])]
X_na=X_na[pd.isna(X_na['Artist End Date'])]
X_na.drop(columns=['Artist Begin Date', 'Artist End Date'], inplace=True)
X_na=X_na.dropna(axis=0)
length=[]
width=[]
for index, row in X_na.iterrows():
    l1 = re.findall('\((.*?) cm', row[7])
    if len(l1) == 1:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", l1[0])
        if len(nums) == 2:
            X_na.loc[index, 'Dimensions'] = float(nums[0]) * float(nums[1])
            length.append(float(nums[0]))
            width.append(float(nums[1]))
        else:
            X_na.drop(index, inplace=True)
    else:
        X_na.drop(index, inplace=True)
X_na['Length']=length
X_na['Width']=width
max_begin2=max(X_na['Object Begin Date'])-min(X_na['Object Begin Date'])
max_end2=max(X_na['Object End Date'])-min(X_na['Object End Date'])
max_dim2=max(X_na['Dimensions'])-min(X_na['Dimensions'])
max_len2=max(X_na['Length'])-min(X_na['Length'])
max_wid2=max(X_na['Width'])-min(X_na['Width'])
mydict_name2={}
mydict_medium2={}
mydict_cl2={}
mydict_class2={}
X_na['Department'] = X_na['Department'].astype('category')
X_na['Object Name'] = X_na['Object Name'].astype('category')
X_na['Medium'] = X_na['Medium'].astype('category')
X_na['Credit Line'] = X_na['Credit Line'].astype('category')
X_na['Classification'] = X_na['Classification'].astype('category')
X_na['depart_cat'] = X_na['Department'].cat.codes
X_na['name_cat'] = X_na['Object Name'].cat.codes
X_na['med_cat'] = X_na['Medium'].cat.codes
X_na['cl_cat'] = X_na['Credit Line'].cat.codes
X_na['class_cat'] = X_na['Classification'].cat.codes
delimiters = ' and ', ' or ', ' ', ',', ';', '&', '(?)', '(', ')', '/', '|', '.'
regexPattern = '|'.join(map(re.escape, delimiters))
for idx, item in enumerate(X_na['name_cat']):
    mydict_name2[item]=list(filter(lambda a: a != '', re.split(regexPattern, X_na['Object Name'][idx])))
for idx, item in enumerate(X_na['med_cat']):
    mydict_medium2[item]=list(filter(lambda a: a != '', re.split(regexPattern, X_na['Medium'][idx])))
for idx, item in enumerate(X_na['cl_cat']):
    delims = ', ', '; '
    rePattern = '|'.join(map(re.escape, delims))
    mydict_cl2[item]=list(filter(lambda a: a != '', re.split(rePattern, X_na['Credit Line'][idx])))
for idx, item in enumerate(X_na['class_cat']):
    mydict_class2[item]=list(filter(lambda a: a != '', re.split(regexPattern, X_na['Classification'][idx])))
    mydict_class2[item]=list(map(lambda x: x.split('-')[0], mydict_class2.get(item)))
X_na.drop(columns=['Department', 'Object Name', 'Medium', 'Credit Line', 'Classification'], inplace=True)
sub.drop(columns=['Artist Begin Date', 'Artist End Date'], inplace=True)
def mydist(x, y):
    sc1=0
    sc2=0
    sc3=0
    sc4=0
    sc5=0
    sc6=0
    sc7=0
    sc8=0
    sc9=0
    sc10=0
    sc11=0
    sc12=0
    if x[0] != y[0]:
        sc1 = 1
    if x[1] != y[1]:
        sc2 = 1
    sc3=abs(x[2]-y[2])/max_begin
    sc4=abs(x[3]-y[3])/max_end
    sc5 = abs(x[4]-y[4])/max_dim
    sc6 = abs(x[5]-y[5])/max_len
    sc7 = abs(x[6]-y[6])/max_wid
    if x[7] != y[7]:
        sc8=1
    sc9=cat_dist(set(mydict_name.get(x[8])), set(mydict_name.get(y[8])), 'Object Name')
    sc10= cat_dist(set(mydict_medium.get(x[9])), set(mydict_medium.get(y[9])), 'Medium')
    sc11 = cat_dist(set(mydict_cl.get(x[10])), set(mydict_cl.get(x[10])), 'Credit Line')
    sc12 = cat_dist(set(mydict_class.get(x[11])), set(mydict_class.get(y[11])), 'Classification')
    return sc1+sc2+sc3+sc4+sc5+sc6+sc7+sc8+sc9+sc10+sc11+sc12
def mydist2(x, y):
    sc1=0
    sc2=0
    sc3=0
    sc4=0
    sc5=0
    sc6=0
    sc7=0
    sc8=0
    sc9=0
    sc10=0
    sc11=0
    sc12=0
    if x[0] != y[0]:
        sc1 = 1
    if x[1] != y[1]:
        sc2 = 1
    sc3=abs(x[2]-y[2])/max_begin2
    sc4=abs(x[3]-y[3])/max_end2
    sc5 = abs(x[4]-y[4])/max_dim2
    sc6 = abs(x[5]-y[5])/max_len2
    sc7 = abs(x[6]-y[6])/max_wid2
    if x[7] != y[7]:
        sc8=1
    sc9=cat_dist(set(mydict_name2.get(x[8])), set(mydict_name2.get(y[8])), 'Object Name')
    sc10= cat_dist(set(mydict_medium2.get(x[9])), set(mydict_medium2.get(y[9])), 'Medium')
    sc11 = cat_dist(set(mydict_cl2.get(x[10])), set(mydict_cl2.get(x[10])), 'Credit Line')
    sc12 = cat_dist(set(mydict_class2.get(x[11])), set(mydict_class2.get(y[11])), 'Classification')
    return sc1+sc2+sc3+sc4+sc5+sc6+sc7+sc8+sc9+sc10+sc11+sc12
l1=[21,8,12,15,9,11,20]
l2=[9,22,11,16,6,6,9,5,17,10,8,16,7]
sp.stats.ranksums(l1,l2)
RanksumsResult(statistic=1.2282647202130073, pvalue=0.21934761016862558)
sp.stats.ranksums(np.array(df['Dimensions']), np.array(df2['Dimensions']))
RanksumsResult(statistic=5.524310971077592, pvalue=3.307811462120068e-08)
np.median(np.array(df['Dimensions']))
691.67
np.median(np.array(df2['Dimensions']))
170.81
np.mean(np.array(df['Dimensions']))
1159.1737500000002
np.mean(np.array(df2['Dimensions']))
1237.455437323944
n, bins, patches = plt.hist(df['Dimensions'], 100, facecolor='blue')
plt.show()
n, bins, patches = plt.hist(df2['Dimensions'], 100, facecolor='blue')
plt.show()
Test Is Public Domain
count = np.array([79, 108])
nobs = np.array([96, 142])
stat, pval = proportions_ztest(count, nobs)
pval
0.2501178231151471
sp.stats.ranksums(np.array(df['Object Begin Date']), np.array(df2['Object Begin Date']))
RanksumsResult(statistic=5.7123889544597715, pvalue=1.1140110866430812e-08)
sp.stats.ranksums(np.array(df['Object End Date']), np.array(df2['Object End Date']))
RanksumsResult(statistic=4.681798372049257, pvalue=2.843691115489764e-06)
n, bins, patches = plt.hist(df['Object Begin Date'], 100, facecolor='blue')
plt.show()
n, bins, patches = plt.hist(df2['Object Begin Date'], 100, facecolor='blue')
plt.show()
n, bins, patches = plt.hist(df['Object End Date'], 100, facecolor='blue')
plt.show()
n, bins, patches = plt.hist(df['Object End Date'], 100, facecolor='blue')
plt.show()
n, bins, patches = plt.hist(df2['Object End Date'], 100, facecolor='blue')
plt.show()
sp.stats.ranksums(np.array(df['Length']), np.array(df2['Length']))
RanksumsResult(statistic=5.182699940036491, pvalue=2.1869672333507455e-07)
sp.stats.ranksums(np.array(df['Width']), np.array(df2['Width']))
RanksumsResult(statistic=4.8871488232930655, pvalue=1.0230675898872156e-06)
'''