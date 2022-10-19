import os
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForMaskedLM
from torch.nn import functional as F
import torch
import numpy as np
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Calculates surprisal and other \
                                    metrics (in development) of transformers language models')

    parser.add_argument('--stimuli', '-i', type=str,
                        help='stimuli to test')
    parser.add_argument('--stimuli_list', '-ii', type=str,
                        help='path to file containing list of stimulus files to test')
    parser.add_argument('--output_directory','-o', type=str, required = True,
                        help='output directory')
    parser.add_argument('--primary_decoder','-d', type=str, default='masked',
                        help='for models with both masked and causal versions, determine which to use (default is masked)')
    parser.add_argument('--model','-m', type=str,
                        help='select a model to use')
    parser.add_argument('--model_list','-mm', type=str,
                        help='path to file with a list of models to run')
    parser.add_argument('--task', '-t', type=str,
                        help='metric to caclulate')
    parser.add_argument('--task_list', '-tt', type=str,
                        help='path to file with list of metrics to caclulate')
    parser.add_argument('--following_context', '-f', action="store_true", default=False,
                        help='whether or not consider the following context with masked language models (default is False)')
    parser.add_argument('--use_cpu', '-cpu', action="store_true", default=False,
                        help='use CPU for models even if CUDA is available')

    args = parser.parse_args()
    return args

def process_args(args):
    try:
        output_directory = args.output_directory
    except:
        print("Error: Please specify a valid output directory.")

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            print("Error: Cannot create output directory (Note: output directory does not already exist).")
        
    try:
        primary_decoder = args.primary_decoder
        assert primary_decoder=="causal" or primary_decoder=="masked"
    except:
        print("Error: Please select either 'causal' or 'masked' for primary decoder argument.")

    try:
        include_following_context = args.following_context
        assert type(include_following_context)==bool
    except:
        print("Error: 'following_context' argument must be Boolean.")
    
    try:
        cpu = args.use_cpu
        assert type(cpu)==bool
    except:
        print("Error: 'use_cpu' argument must be Boolean.")

    if args.model_list:
        try:
            assert os.path.exists(args.model_list)
            with open(args.model_list, "r") as f:
                model_list = f.read().splitlines()
        except:
            print("Error: 'model_list' argument does not have a valid path. Trying to use individual specified model.")
            try:
                assert args.model
                model_list = [args.model]
            except:
                print("Error: No model specified")
    else:
        try:
            assert args.model
            model_list = [args.model]
        except:
            print("Error: No model specified")        



    if args.task_list:
        try:
            assert os.path.exists(args.task_list)
            with open(args.task_list, "r") as f:
                metric_list = f.read().splitlines()
        except:
            print("Error: 'metric_list' argument does not have a valid path. Trying to use individual specified metric.")
            try:
                assert args.task
                metric_list = [args.task]
            except:
                print("Error: No metric specified")
    else:
        try:
            assert args.task
            metric_list = [args.task]
        except:
            print("Error: No metric specified")    
            
            
    if args.stimuli_list:
        try:
            assert os.path.exists(args.stimuli_list)
            with open(args.stimuli_list, "r") as f:
                stimulus_file_list = f.read().splitlines()
        except:
            print("Error: 'stimuli_list' argument does not have a valid path. Trying to use individual stimulus set.")
            try:
                assert args.stimuli
                stimulus_file_list = [args.stimuli]
            except:
                print("Error: No stimuli specified")
    else:
        try:
            assert args.stimuli
            stimulus_file_list = [args.stimuli]
        except:
            print("Error: No stimuli specified")  
                
    return(output_directory,primary_decoder,include_following_context,model_list,metric_list,stimulus_file_list,cpu)  

def create_and_run_models(model_list,stimulus_file_list,metric_list,primary_decoder,output_directory,include_following_context,cpu):
    if primary_decoder == "masked":
        for model_name in model_list:
            
            model_name_cleaned = model_name.replace("/","-")
            
            if 'tokenizer' in locals():
                del(tokenizer)
            
            if 'model' in locals():
                del(model)
                
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if (not tokenizer.bos_token) and (tokenizer.cls_token):
                    tokenizer.bos_token = tokenizer.cls_token
                if (not tokenizer.eos_token) and (tokenizer.sep_token):
                    tokenizer.eos_token = tokenizer.sep_token

                tokenizer.add_tokens(["[!StimulusMarker!]"," [!StimulusMarker!]"])

            except:
                print("Cannot create a tokenizer for model {0}".format(model_name))
                
            try:
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                model_type = "masked"
            except:
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_name,is_decoder=True)
                    model_type = "causal"
                except:
                    print("Model {0} is not a masked or causal language model. This is not supported".format(model_name))
            try:
                assert model and tokenizer
                if model and tokenizer:
                    try:
                        if model_type=="causal":
                            process_stims_causal(model.to("cuda" if (torch.cuda.is_available() and not cpu) else "cpu"),tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context)
                        elif model_type=="masked":
                            process_stims_masked(model.to("cuda" if (torch.cuda.is_available() and not cpu) else "cpu"),tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context)
                    except:
                        print("Cannot run either a masked or causal form of {0}".format(model_name))
            except:
                print("Cannot run experiment without both a tokenizer for and a causal or masked form of {0}".format(model_name))
    
    elif primary_decoder == "causal":
        for model_name in model_list:
            
            model_name_cleaned = model_name.replace("/","-")

            if 'tokenizer' in locals():
                del(tokenizer)
            
            if 'model' in locals():
                del(model)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if (not tokenizer.bos_token) and (tokenizer.cls_token):
                    tokenizer.bos_token = tokenizer.cls_token
                if (not tokenizer.eos_token) and (tokenizer.sep_token):
                    tokenizer.eos_token = tokenizer.sep_token

                tokenizer.add_tokens(["[!StimulusMarker!]"," [!StimulusMarker!]"])

            except:
                print("Cannot create a tokenizer for model {0}".format(model_name))
                
            try:            
                model = AutoModelForCausalLM.from_pretrained(model_name,is_decoder=True)
                model_type = "causal"
                if "Masked" in model.config.architectures[0]:
                    model_type = "causal_mask"                    
            except:
                try:
                    model = AutoModelForMaskedLM.from_pretrained(model_name)
                    model_type = "masked"
                except:
                    print("Model {0} is not a causal or masked language model. This is not supported".format(model_name))
            try:
                assert model and tokenizer
                if model and tokenizer:
                    try:
                        if model_type=="causal":
                            process_stims_causal(model.to("cuda" if (torch.cuda.is_available() and not cpu) else "cpu"),tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context)
                        elif model_type=="masked":
                            process_stims_masked(model.to("cuda" if (torch.cuda.is_available() and not cpu) else "cpu"),tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context)
                        elif model_type=="causal_mask":
                            process_stims_causal_mask(model.to("cuda" if (torch.cuda.is_available() and not cpu) else "cpu"),tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context)
                                            
                    except:
                        print("Cannot run either a causal or masked form of {0}".format(model_name))
            except:
                print("Cannot run experiment without both a tokenizer for and a causal or masked form of {0}".format(model_name))  

                  
def process_stims_causal(model,tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context):
    for i in range(len(stimulus_file_list)):
        stimuli_name = stimulus_file_list[i].split('/')[-1].split('.')[0] 
        
        if "surprisal" in metric_list:
            filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".causal.output"
            with open(filename,"w") as f:
                f.write("FullSentence\tSentence\tTargetWords\tSurprisal\tNumTokens\n")
                
        with open(stimulus_file_list[i],'r') as f:
            stimulus_list = f.read().splitlines() 
        for j in range(len(stimulus_list)):
            try:
                stimulus = stimulus_list[j]
                stimulus_spaces = stimulus.replace("*", "[!StimulusMarker!]")
                stimulus_spaces = stimulus_spaces.replace(" [!StimulusMarker!]", "[!StimulusMarker!] ")
                encoded_stimulus = tokenizer.encode(stimulus_spaces)
                
                if (len(tokenizer.tokenize("a[!StimulusMarker!]"))==2):
                    dummy_var_idxs = np.where((np.array(encoded_stimulus)==tokenizer.encode("[!StimulusMarker!]")[-1]) | (np.array(encoded_stimulus)==tokenizer.encode("a[!StimulusMarker!]")[-1]))[0]
                    preceding_context = encoded_stimulus[:dummy_var_idxs[0]]
                    if (len(preceding_context)==0) or (not ((preceding_context[0]==tokenizer.bos_token_id) or (preceding_context[0]==tokenizer.eos_token_id))):
                        preceding_context = [tokenizer.bos_token_id] + preceding_context
                    target_words = encoded_stimulus[dummy_var_idxs[0]+1:dummy_var_idxs[1]]
                    following_words = encoded_stimulus[dummy_var_idxs[1]+1:]
                    if "surprisal" in metric_list:
                        get_surprisal_causal(model,tokenizer,preceding_context,following_words,target_words,stimuli_name,model_name_cleaned,output_directory,stimulus)
            except:
                print("Problem with stimulus on line {0}: {1}\n".format(str(j+1),stimulus_list[j]))


def get_surprisal_causal(model,tokenizer,preceding_context,following_words,target_words,stimuli_name,model_name_cleaned,output_directory,stimulus):
    filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".causal.output"
    current_context = copy.deepcopy(preceding_context)
    all_probabilities = []
    for i in range(len(target_words)):
        current_target = target_words[i]
        input = torch.LongTensor([current_context]).to(model.device)
        with torch.no_grad():
            next_token_logits = model(input, return_dict=True).logits[:, -1, :]
        probs = F.softmax(next_token_logits,dim=-1)
        probability = probs[0,current_target]
        current_context.append(current_target)
        all_probabilities.append(probability.item())
    all_probabilities = np.array(all_probabilities)
    num_tokens = len(all_probabilities)
    sum_surprisal = np.sum(-np.log2(all_probabilities))
    sentence = tokenizer.decode(preceding_context[1:]+target_words)
    full_sentence = tokenizer.decode(preceding_context[1:]+target_words+following_words)
    target_string = tokenizer.decode(target_words)
    with open(filename,"a") as f:
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
            stimulus.replace("*",""),
            sentence,
            target_string,
            sum_surprisal,
            num_tokens
        ))

        
def process_stims_masked(model,tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context):
    for i in range(len(stimulus_file_list)):
        stimuli_name = stimulus_file_list[i].split('/')[-1].split('.')[0] 
        
        if "surprisal" in metric_list:
            filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".masked.output"
            with open(filename,"w") as f:
                f.write("FullSentence\tSentence\tTargetWords\tSurprisal\tNumTokens\n")
                 
        with open(stimulus_file_list[i],'r') as f:
            stimulus_list = f.read().splitlines() 
        for j in range(len(stimulus_list)):  
            try:          
                stimulus = stimulus_list[j]
                stimulus_spaces = stimulus.replace("*", "[!StimulusMarker!]")
                if (tokenizer.tokenize(" a")[0][0]==tokenizer.tokenize(" b")[0][0]) and (tokenizer.tokenize("a")[0][0]!=tokenizer.tokenize("b")[0][0]):
                    stimulus_spaces = stimulus_spaces.replace(" [!StimulusMarker!]", "[!StimulusMarker!] ")
                else:
                    stimulus_spaces = stimulus_spaces.replace("[!StimulusMarker!]", "[!StimulusMarker!] ")
                    stimulus_spaces = stimulus_spaces.replace(" [!StimulusMarker!]", "[!StimulusMarker!]")
                encoded_stimulus = tokenizer.encode(stimulus_spaces)[1:-1]
                
                if (len(tokenizer.tokenize("a[!StimulusMarker!]"))==2):
                    dummy_var_idxs = np.where((np.array(encoded_stimulus)==tokenizer.encode("[!StimulusMarker!]")[-2]) | (np.array(encoded_stimulus)==tokenizer.encode("a[!StimulusMarker!]")[-2]))[0]
                    preceding_context = encoded_stimulus[:dummy_var_idxs[0]]
                    if (len(preceding_context)==0) or (not preceding_context[0]==tokenizer.bos_token_id):
                        preceding_context = [tokenizer.bos_token_id] + preceding_context
                    target_words = encoded_stimulus[dummy_var_idxs[0]+1:dummy_var_idxs[1]]
                    following_words = encoded_stimulus[dummy_var_idxs[1]+1:]
                    if "surprisal" in metric_list:
                        get_surprisal_masked(model,tokenizer,preceding_context,following_words,target_words,stimuli_name,model_name_cleaned,output_directory,include_following_context,stimulus)
            except:
                print("Problem with stimulus on line {0}: {1}\n".format(str(j+1),stimulus_list[j]))

def get_surprisal_masked(model,tokenizer,preceding_context,following_words,target_words,stimuli_name,model_name_cleaned,output_directory,include_following_context,stimulus):
    filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".masked.output"
    current_context = copy.deepcopy(preceding_context)
    all_probabilities = []
    for i in range(len(target_words)):
        current_target = target_words[i]
        context_plus_mask = current_context + [tokenizer.mask_token_id]
        if include_following_context==True:
            context_plus_mask = context_plus_mask + following_words
        model_input_list = context_plus_mask+[tokenizer.eos_token_id]
        mask_idx = model_input_list.index(tokenizer.mask_token_id)
        input = torch.LongTensor([model_input_list]).to(model.device)
        with torch.no_grad():
            next_token_logits = model(input, return_dict=True).logits[:, mask_idx, :]
        probs = F.softmax(next_token_logits,dim=-1)
        probability = probs[0,current_target]
        current_context.append(current_target)
        all_probabilities.append(probability.item())
    all_probabilities = np.array(all_probabilities)
    num_tokens = len(all_probabilities)
    sum_surprisal = np.sum(-np.log2(all_probabilities))
    sentence = tokenizer.decode(preceding_context[1:]+target_words)
    full_sentence = tokenizer.decode(preceding_context[1:]+target_words+following_words)
    target_string = tokenizer.decode(target_words)
    with open(filename,"a") as f:
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
            stimulus.replace("*",""),
            sentence,
            target_string,
            sum_surprisal,
            num_tokens
        ))
        
def process_stims_causal_mask(model,tokenizer,stimulus_file_list,metric_list,model_name_cleaned,output_directory,include_following_context):
    for i in range(len(stimulus_file_list)):
        stimuli_name = stimulus_file_list[i].split('/')[-1].split('.')[0] 
        
        if "surprisal" in metric_list:
            filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".causal_mask.output"
            with open(filename,"w") as f:
                f.write("FullSentence\tSentence\tTargetWords\tSurprisal\tNumTokens\n")
                 
        with open(stimulus_file_list[i],'r') as f:
            stimulus_list = f.read().splitlines() 
        for j in range(len(stimulus_list)): 
            try:           
                stimulus = stimulus_list[j]
                stimulus_spaces = stimulus.replace("*", "[!StimulusMarker!]")
                if (tokenizer.tokenize(" a")[0][0]==tokenizer.tokenize(" b")[0][0]) and (tokenizer.tokenize("a")[0][0]!=tokenizer.tokenize("b")[0][0]):
                    stimulus_spaces = stimulus_spaces.replace(" [!StimulusMarker!]", "[!StimulusMarker!] ")
                else:
                    stimulus_spaces = stimulus_spaces.replace("[!StimulusMarker!]", "[!StimulusMarker!] ")
                    stimulus_spaces = stimulus_spaces.replace(" [!StimulusMarker!]", "[!StimulusMarker!]")
                encoded_stimulus = tokenizer.encode(stimulus_spaces)[1:-1]
                
                if (len(tokenizer.tokenize("a[!StimulusMarker!]"))==2):
                    dummy_var_idxs = np.where((np.array(encoded_stimulus)==tokenizer.encode("[!StimulusMarker!]")[-2]) | (np.array(encoded_stimulus)==tokenizer.encode("a[!StimulusMarker!]")[-2]))[0]
                    preceding_context = encoded_stimulus[:dummy_var_idxs[0]]
                    if (len(preceding_context)==0) or (not preceding_context[0]==tokenizer.bos_token_id):
                        preceding_context = [tokenizer.bos_token_id] + preceding_context
                    target_words = encoded_stimulus[dummy_var_idxs[0]+1:dummy_var_idxs[1]]
                    following_words = encoded_stimulus[dummy_var_idxs[1]+1:]
                    
                    if "surprisal" in metric_list:
                        get_surprisal_causal_mask(model,tokenizer,preceding_context,following_words,target_words,stimuli_name,model_name_cleaned,output_directory,include_following_context,stimulus)
            except:
                print("Problem with stimulus on line {0}: {1}\n".format(str(j+1),stimulus_list[j]))

def get_surprisal_causal_mask(model,tokenizer,preceding_context,following_words,target_words,stimuli_name,model_name_cleaned,output_directory,include_following_context,stimulus):
    filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".causal_mask.output"
    current_context = copy.deepcopy(preceding_context)
    all_probabilities = []
    for i in range(len(target_words)):
        current_target = target_words[i]
        context_plus_mask = current_context + [tokenizer.mask_token_id]
        if include_following_context==True:
            context_plus_mask = context_plus_mask + following_words
        model_input_list = context_plus_mask+[tokenizer.eos_token_id]
        mask_idx = model_input_list.index(tokenizer.mask_token_id)
        input = torch.LongTensor([model_input_list]).to(model.device)
        with torch.no_grad():
            next_token_logits = model(input, return_dict=True).logits[:, mask_idx, :]
        probs = F.softmax(next_token_logits,dim=-1)
        probability = probs[0,current_target]
        current_context.append(current_target)
        all_probabilities.append(probability.item())
    all_probabilities = np.array(all_probabilities)
    num_tokens = len(all_probabilities)
    sum_surprisal = np.sum(-np.log2(all_probabilities))
    sentence = tokenizer.decode(preceding_context[1:]+target_words)
    full_sentence = tokenizer.decode(preceding_context[1:]+target_words+following_words)
    target_string = tokenizer.decode(target_words)
    with open(filename,"a") as f:
        f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
            stimulus.replace("*",""),
            sentence,
            target_string,
            sum_surprisal,
            num_tokens
        ))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    output_directory,primary_decoder,include_following_context,model_list,metric_list,stimulus_file_list,cpu = process_args(args)
    create_and_run_models(model_list,stimulus_file_list,metric_list,primary_decoder,output_directory,include_following_context,cpu)

if __name__ == "__main__":
    main()
