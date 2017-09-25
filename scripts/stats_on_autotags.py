#############################################################################
#   This file was adapted from the SNLI version written by Sam Bowman.      #
#   Current version created by A. Williams                                  #
#############################################################################

from collections import defaultdict, Counter
import random
import re
import os
import operator
import json
import sys
import unicodecsv as csv


#############################################################################
#   This file takes as input MNLI data, and model output files and assigns  #
#   autotags, outputs percentages etc.                                      #
#   You will need: paths to corpus files and model file to run this script  #
#############################################################################

tags_to_results = defaultdict(list)

#######################################
#     relevant function defs here     #
#######################################

def log(tag, is_correct, label):
    tags_to_results[tag].append((is_correct, label))

def find_1st_verb(str1):  #find ptb verb codings for first verb from root of sentence
    findy=str1.find('(VB')
    if findy >0:
        return str1[findy:].split()[0]
    else: 
        return ''

def tense_match(str1,str2):
    result=find_1st_verb(str1)
    if len(result)>0:
        findy2=str2.find(result)
        return findy2>0
    else:
        return False

def printStats(genrename,modelname,corpusname):
    print '***********************************'+ genrename + '*****************************************************************'
    os.chdir(os.getcwd()+'/'+str(modelname))
    os.getcwd()
    with open(str(modelname)+'_'+str(corpusname)+'_'+str(genrename)+'.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            
            for tag in sorted(tags_to_results):
                correct = len([result[0] for result in tags_to_results[tag] if result[0]])
                counts = Counter([result[1] for result in tags_to_results[tag]])
                best_label, best_count = max(counts.iteritems(), key=operator.itemgetter(1))

                attempted = len(tags_to_results[tag])
                baseline = float(best_count) / attempted 

                acc = float(correct)/attempted
                totalpercent= float(attempted)/i
                print tag, "\t", correct, "\t", attempted, "\t", acc, "\t", baseline, "\t", best_label, "\t", totalpercent 

                writer.writerow([tag, correct, attempted, acc, baseline,  best_label, totalpercent])
            print ''
            print 'writing ' +str(modelname)+'_'+str(genrename) +' to file'
    print 'made ' + str(corpusname) + '_'+ str(modelname) + '_' + str(genrename) + ' results file tab-sep'
    
    print str(i) + ' examples found'   
    os.chdir('/Users/Adina/Documents/MultiGenreNLI/')
    print '*******************************************************************************************************'

# def model_correct_or_not(pairid, label, filename):
#     with open(filename, 'rbU') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             print row
#             pID= row[0]
#             model_guess=row[1]
#             if pairid==pID:
#                 if label ==model_guess:
#                     correct = True 
#                 else:
#                     correct = False

def make_dict(filename,dictname):
    mydict = {}
    reader = csv.reader(open(filename, "rb"))
    for rows in reader:
        k = rows[0]
        v = rows[1]
        mydict[k] = v
    dictname.update(mydict)
    print len(dictname)
    print 'model dict created'

######################
#                    #
# get stats by genre #
#                    #
######################

def get_stats(row,model_dict):
    global i
    label = row[0]
    if label in ["entailment", "contradiction", "neutral"]:
        pairid = row[8]
        p1 = row[3]
        p2 = row[4]
        b1 = row[1]
        b2 = row[2]
        t1 = row[5]
        t2 = row[6]
        genre = row[9]
        b1t = b1.split()
        b2t = b2.split()
        sb1t = set(b1t)
        sb2t = set(b2t)
        parses = p1 + " " + p2
        if model_dict[pairid]==label:
            correct=True. # this needs to be supplied from the model outputs
        else:
            correct=False
        i+=1
        log("label-" + label, correct, label)
        
##################
#   hand chosen  #
#   NEG/DET      #
##################

        if "n't" in parses or "not" in parses or "none" in parses or "never" in parses or "neither" in parses or "nor" in parses:  # add in un- and non- :/
            log('neg-all', correct, label)
            if ("n't" in p2 or "not" in p2 or "none" in p2 or "never" in p2 or "neither" in p2 or "nor" in p2) and not ("n't" in p1 or "not" in p1 or "none" in p1 or "never" in p1 or "neither" in p1 or "nor" in p1):
                log('neg-hyp-only', correct, label)


        if "a" in parses or "the" in parses or "these" in parses or "this" in parses or "those" in parses or "that" in parses: 
            log('det-all', correct, label)
            if ("a" in p2 or "the" in p2 or "these" in p2 or "this" in p2 or "those" in p2 or "that" in p2) and not ("a" in p1 or "the" in p1 or "these" in p1 or "this" in p1 or "those" in p1 or "that" in p1):
                log('det-hyp-only', correct, label)

##################
#    PTB TAGS    #
##################
        for key in ptbtags:
            if key in parses:
                log(ptbtags[key]+'_ptb_all', correct, label)
            if (key in p2) and not (key in p1):
                log(ptbtags[key]+'_ptb_hyp_only', correct, label)

        if ("(NNS"  in p2) and ("(NNP" in p1):
            log('plural-premise-sing-hyp_ptb', correct, label)
        if ("(NNP"  in p2) and ("(NNS" in p1):
            log('plural-hyp-sing-premise_ptb', correct, label)

        if tense_match(p1,p2):
            log('tense_match', correct, label)

###################
#  interjects &   #
#  foreign words  #
###################

        if "(UH" in parses: 
            log('interject-all_ptb', correct, label)
            if ("(UH"  in p2) and not ("(UH" in p1):
                log('interject-hyp-only_ptb', correct, label)

        if "(FW" in parses: 
            log('foreign-all_ptb', correct, label)
            if ("(FW"  in p2) and not ("(FW" in p1):
                log('foreign-hyp-only_ptb', correct, label)

###################
#  PTB modifiers  #
###################

        if "(JJ" in parses: 
            log('adject-all_ptb', correct, label)
            if ("(JJ"  in p2) and not ("(JJ" in p1):
                log('adject-hyp-only_ptb', correct, label)

        if "(RB" in parses: 
            log('adverb-all_ptb', correct, label)
            if ("(RB"  in p2) and not ("(RB" in p1):
                log('adverb-hyp-only_ptb', correct, label)

        if "(JJ" in parses or "(RB" in parses: 
            log('adj/adv-all_ptb', correct, label)
            if ("(JJ"  in p2 or "(RB" in p2) and not ("(JJ" in p1 or "(RB" in p1):
                log('adj/adv-hyp-only_ptb', correct, label)
# modifiers are good examples of how additions/subtractions result in neutral

# if hyp (and premise) have -er, -est, adjectives,z or adverbs in them
        if "(RBR" in parses or "(RBS" in parses or "(JJR" in parses or "(JJS" in parses: 
            log('er-est-all_ptb', correct, label)
            if ("(RBR"  in p2 or "(RBS" in p2 or "(JJR" in p2 or "(JJS" in p2) and not ("(RBR" in p1 or "(RBS" in p1 or "(JJR" in p1 or "(JJS" in p1):
                log('er-est-hyp-only_ptb', correct, label)

#########################
#  S-Root, length etc.  #
#########################

        s1 = p1[0:8] == "(ROOT (S"
        s2 = p2[0:8] == "(ROOT (S" 
        if s1 and s2:
            log('syn-S-S', correct, label)
        elif s1 or s2:
            log('syn-S-NP', correct, label)
        else:
            log('syn-NP-NP', correct, label)

        prem_len = len([word for word in b2.split() if word != '(' and word != ')'])
        if prem_len < 11:
            log('len-0-10', correct, label)
        elif prem_len < 15:
            log('len-11-14', correct, label)
        elif prem_len < 20:
            log('len-15-19', correct, label)
        else:
            log('len-20+', correct, label)

        if sb1t.issubset(sb2t):
            log('token-ins-only', correct, label)
        elif sb2t.issubset(sb1t):
            log('token-del-only', correct, label)


        if len(sb1t.difference(sb2t)) == 1 and len(sb2t.difference(sb1t)) == 1:
            log('token-single-sub-or-move', correct, label)

        if len(sb1t.union(sb2t)) > 0:
            overlap = float(len(sb1t.intersection(sb2t)))/len(sb1t.union(sb2t)) 
            if overlap > 0.6:
                log('overlap-xhigh', correct, label)
            elif overlap > 0.37:
                log('overlap-high', correct, label)
            elif overlap > 0.23:
                log('overlap-mid', correct, label)
            elif overlap > 0.12:
                log('overlap-low', correct, label)
            else:
                log('overlap-xlow', correct, label)
        else: 
            log('overlap-empty', correct, label) 

##############
#   GREPing  #
##############


        for keyphrase in ["much", "enough", "more", "most", "every", "each", "less", "least", "no", "none", "some", "all", "any", "many", "few", "several"]:  # get a list from Anna's book, think more about it
            if keyphrase in p2 or keyphrase in p1:
                log('template-quantifiers', correct, label)
                break

        for keyphrase in ["know", "knew", "believe", "understood", "understand", "doubt", "notice", "contemplate", "consider", "wonder", "thought", "think", "suspect", "suppose", "recognize",  "recognise", "forgot", "forget", "remember",  "imagine", "meant", "agree", "mean",  "disagree", "denied", "deny", "promise"]:
            if keyphrase in p2 or keyphrase in p1:
                log('template-beliefVs', correct, label)
                break

        for keyphrase in ["love", "hate", "dislike", "annoy", "angry",  "happy", "sad", "bliss", "blissful", "depress","terrified","terrify", "scare", "amuse", "suprise", "guilt", "fear", "afraid", "startle",  "confuse", "baffle", "frustrate", "enfuriate", "rage", "befuddle", "fury", "furious", "elated", "elation", "joy", "joyous", "joyful", "enjoy", "relish"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-psychpreds', correct, label)
                    break

        for keyphrase in ['if']:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-if', correct, label)
                    break

        for keyphrase in ["May I", "Mr.", "Mrs." "Ms.", "Dr.", "excuse me", "Excuse me", "pardon me", "sorry", "Sorry", "I'm sorry", "I am sorry", "Pardon me", 'please', 'thank', 'thanks', 'Thanks', 'Thank', 'Please', "you're welcome", "You're welcome", "much obliged", "Much obliged"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-polite', correct, label)
                    break

        for keyphrase in ["time", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "morning", "night", "tomorrow", "yesterday", "evening", "week", "weeks", "hours", "minutes", "seconds" "hour", "days", "years", "decades", "lifetime", "lifetimes", "epoch", "epochs", "day", "recent", "recently", "habitually", "whenever", "during", "while", "before", "after", "previously", "again", "often", "repeatedly", "frequently", "dusk", "dawn", "midnight", "afternoon", "when", "daybreak", "later", "earlier", "month", "year", "decade", "biweekly", "millenium", "midday", "daily", "weekly", "monthly", "yearly", "hourly", "fortnight", "now", "then"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-timeterms', correct, label)
                    break

        for keyphrase in ["too", "anymore", "also", "as well", "again", "no longer", "start", "started", "starting", "stopping", "stop", "stopped", "regretting", "regret", "regretted", "realizing", "realize", "realized", "aware", "manage", "managed", "forgetting", "forget", "forgot", "began", "begin", "finish", "finished", "finishing", "ceasing", "cease", "ceased", "enter", "entered", "entering", "leaving", "leave", "left", "carry on", "carried on", "return", "returned", "returning", "restoring", "restore", "restored", "repeat", "repeated", "repeating", "another", "only", "coming back", "come back", "came back"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-presupptrigs', correct, label)
                    break

        for keyphrase in ["although", "but", "yet", "despite", "however", "However", "Although", "But", "Yet", "Despite", "therefore", "Therefore", "Thus", "thus"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-convo-pivot', correct, label)
                    break                          

        for keyphrase in ["weight", "height", "age", "width", "length", "mother", "father", "sister", "brother", "aunt", "uncle", "cousin", "husband", "wife", "mom", "dad", "Mom", "Dad", "Mama", "Papa", "mama", "papa", "grandma", "grandpa", "nephew", "niece", "widow", "family", "kin", "bride", "spouse"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-relNs', correct, label)
                    break      

        len(tags_to_results)
    

########################################################################################################################
########################################################################################################################


#################################
#       define things here      #
#       tp run the script!      #
#################################

genrelist = ['oup', 'facetoface', 'letters', 'nineeleven', 'verbatim','government', 'travel', 'fiction', 'telephone', 'slate']
glistm = ['government', 'travel', 'fiction', 'telephone', 'slate']
glistmm = ['oup', 'facetoface', 'letters', 'nineeleven', 'verbatim']

#################################
#     paths to corpus files     #
#################################

dev_m_path = '../MultiGenreNLI/mnli_0.9_again/multinli_0.9_dev_matched.txt'
dev_mm_path = '../MultiGenreNLI/mnli_0.9_again/multinli_0.9_dev_mismatched.txt'
#test_m_path = '../MultiGenreNLI/mnli_0.9/multinli_0.9_test_matched.txt'
#test_mm_path = '../MultiGenreNLI/mnli_0.9/multinli_0.9_test_mismatched.txt'
dev_path = '../MultiGenreNLI/mnli_0.9_again/multinli_0.9_dev_all_new.txt' # concatenated match and mismatch
#test_path = '../MultiGenreNLI/mnli_0.9/multinli_0.9_test_all.txt' # concatenated match and mismatch
#snli_test_path = '/../MultiGenreNLI/snli_1.0/snli_1.0_test.txt'
snli_dev_path = '../MultiGenreNLI/snli_1.0/snli_1.0_dev.txt'
#train_path = '../MultiGenreNLI/mnli_0.9/multinli_0.9_train.txt'

#################################
#  paths to model output files  #
#################################

bilstm_m_path ='../MultiGenreNLI/devSetResults/bilstm_multi09_snli_mean_dev_matched_predictions.csv'
bilstm_mm_path ='../MultiGenreNLI/devSetResults/bilstm_multi09_snli_mean_dev_mismatched_predictions.csv'
bilstm_all_path='../MultiGenreNLI/devSetResults/bilstm_multi09_snli_mean_dev_all_predictions.csv'

cbow_m_path ='../MultiGenreNLI/devSetResults/cbow_multi09_snli_dev_matched_predictions.csv'
cbow_mm_path ='../MultiGenreNLI/devSetResults/cbow_multi09_snli_dev_mismatched_predictions.csv'
cbow_all_path ='/Users/Adina/Documents/MultiGenreNLI/devSetResults/cbow_multi09_snli_dev_all_predictions.csv'

ebim_m_path ='../MultiGenreNLI/devSetResults/ebim_multi09_snli_dev_matched_predictions.csv'
ebim_mm_path ='../MultiGenreNLI/devSetResults/ebim_multi09_snli_dev_mismatched_predictions.csv'
ebim_all_path='../MultiGenreNLI/devSetResults/ebim_multi09_snli_dev_all_predictions.csv'

bilstm_all_dict={}
ebim_all_dict={}
cbow_all_dict={}
make_dict(bilstm_all_path,bilstm_all_dict)
make_dict(cbow_all_path,cbow_all_dict)
make_dict(ebim_all_path,ebim_all_dict)

bilstm_m_dict={}
ebim_m_dict={}
cbow_m_dict={}
make_dict(bilstm_m_path,bilstm_m_dict)
make_dict(cbow_m_path,cbow_m_dict)
make_dict(ebim_m_path,ebim_m_dict)

bilstm_mm_dict={}
ebim_mm_dict={}
cbow_mm_dict={}
make_dict(bilstm_mm_path,bilstm_mm_dict)
make_dict(cbow_mm_path,cbow_mm_dict)
make_dict(ebim_mm_path,ebim_mm_dict)

print 'all model guess dict(s) made'


# dict of interesting ptb tags 

ptbtags={"(MD":"modal","(W":"WH","(CD":"card","(PRP":"pron","(EX":"exist","(IN":"prep","(POS":"'s"} 

#####################################
#     What model you looking at?    #
# Be careful, files will be created #
# based on your naming conventions! #
# Think first! Also, 'using_dict'   #
# should reflect which corpus you   #
# are using, because model pairIDs  #
# need to match with the corpus     #
# predictions. Good Luck!           # 
#####################################

using_dict= cbow_mm_dict # e.g. bilstm_all_snli_dict
using_dict_name='CBOW_mismatch_results' # adjust this
corpus_name = 'MNLI' # 'MNLI' or "SNLI"
path = dev_mm_path # check to be sure this matches the corpus name, underspecified is MNLI, and the m/mm/all of the using_dict
print 'I am making ' + str(using_dict_name) + ' using ' +str(corpus_name) 

########################################################################################################################
########################################################################################################################

# this will create a folder for the model in main dir, unless it already exists

try:
    os.mkdir(corpus_name)
except OSError:
    if not os.path.isdir(corpus_name):
        raise
print "made directory folder named after corpus (if it didn't exist already)"

os.chdir('/Users/Adina/Documents/MultiGenreNLI/'+corpus_name)

try:
    os.mkdir(using_dict_name)
except OSError:
    if not os.path.isdir(using_dict_name):
        raise
print "made directory folder named after model (if it didn't exist already)"


#####################################


with open(path, 'rbU') as csvfile:  
    if corpus_name == 'MNLI':
        tags_to_results.clear()
        reader = csv.reader(csvfile, delimiter="\t")
        i=0
        for row in reader:
            get_stats(row,using_dict) 
        print '*******************************************************************************************************'
        print '*******************************************************************************************************'
        print '                                     ' +str(using_dict_name) 
        print '*******************************************************************************************************'
        print '*******************************************************************************************************'
        print ''
        printStats('all',using_dict_name, corpus_name)
        os.chdir('/Users/Adina/Documents/MultiGenreNLI/'+corpus_name)
        j=i
        k=0

        for genre in genrelist:
            tags_to_results.clear()
            csvfile.seek(0) # starts reading at beginning
            reader = csv.reader(csvfile, delimiter="\t")
            i=0
            for row in reader:
                if row[9]==genre:
                    get_stats(row,using_dict)
            print ''
            printStats(genre, using_dict_name, corpus_name)
            os.chdir('/Users/Adina/Documents/MultiGenreNLI/'+corpus_name)
            k += i

        print " total number of exs all: " + str(k) +  ", total number of exs summing genre lists together: " + str(j) 
        # some rows have no gold_label, so this will not be 2k exs per genre
    else:
        tags_to_results.clear()
        reader = csv.reader(csvfile, delimiter="\t")
        i=0
        for row in reader:
            get_stats(row,using_dict) 
        print '*******************************************************************************************************'
        print '*******************************************************************************************************'
        print '                                     ' +str(using_dict_name) 
        print '*******************************************************************************************************'
        print '*******************************************************************************************************'
        print ''
        printStats('all',using_dict_name, corpus_name)
        j=i
        k=0
        print " total number of exs all: " + str(k) +  ", total number of exs summing genre lists together: " + str(j) 
os.chdir('/Users/Adina/Documents/MultiGenreNLI/')
# for tag in sorted(tags_to_results):
#     correct = len([result[0] for result in tags_to_results[tag] if result[0]])
#     counts = Counter([result[1] for result in tags_to_results[tag]])
#     best_label, best_count = max(counts.iteritems(), key=operator.itemgetter(1))

#     attempted = len(tags_to_results[tag])
#     baseline = float(best_count) / attempted 

#     acc = float(correct)/attempted
#     totalpercent= float(attempted)/i

#     print tag, "\t", correct, "\t", attempted, "\t", acc, "\t", baseline, "\t", best_label, "\t", totalpercent 
