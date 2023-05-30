from transformers import PreTrainedTokenizer
import torch
class BpeTokenizer:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
        self.special_tokens_dict = {
            '<unk>': '<unk>',
            '\n': '<|n|>',
            '.': '<|dot|>',
            ',': '<|comma|>',
            '!': '<|exclamation|>',
            '?': '<|question|>',
            '#': '<|hash|>',
            '@': '<|at|>',
            '$': '<|dollar|>',
            '%': '<|percent|>',
            '&': '<|ampersand|>',
            '*': '<|asterisk|>',
            '-': '<|dash|>',
            '+': '<|plus|>',
            '=': '<|equal|>',
            '/': '<|slash|>',
            '\\': '<|backslash|>',
            '|': '<|pipe|>',
            ':': '<|colon|>',
            ';': '<|semicolon|>',
            '(': '<|left_parenthesis|>',
            ')': '<|right_parenthesis|>',
            '[': '<|left_bracket|>',
            ']': '<|right_bracket|>',
            '{': '<|left_curly_brace|>',
            '}': '<|right_curly_brace|>',
            '<': '<|less_than|>',
            '>': '<|greater_than|>',
            '~': '<|tilde|>',
            '`': '<|backtick|>'
        }
        
        self.tokenizer.add_tokens(list(self.special_tokens_dict.values()))
        self.tokenizer.add_special_tokens({'unk_token': '<unk>'})
    
    def encode(self, text):
        # Replace special characters with special tokens
        for char, token in self.special_tokens_dict.items():
            text = text.replace(char, ' ' + token + ' ')
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # Convert tokens to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create a 2D tensor
        tensor = torch.tensor([input_ids]).unsqueeze(0)
        
        return tensor
    
    def decode(self, tensor):
        # Convert tensor to a list of input IDs
        input_ids = tensor.tolist()[0]
        
        # Convert input IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Join tokens and remove special tokens
        text = ' '.join(tokens)
        
        # Replace special tokens with special characters
        for token, char in self.special_tokens_dict.items():
            text = text.replace(token, char)
        
        return text
