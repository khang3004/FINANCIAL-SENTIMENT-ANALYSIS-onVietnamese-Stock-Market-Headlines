"""
Vietnamese Text Preprocessor for Financial Sentiment Analysis

This module handles Vietnamese text preprocessing including:
- Unicode normalization
- Word segmentation using underthesea
- Part-of-speech tagging
- Special word handling (negation, etc.)
- Stock code removal
- Time and number removal
"""

import re
import regex
from typing import List, Dict, Optional
from underthesea import word_tokenize, pos_tag, sent_tokenize


class VietnameseTextPreprocessor:
    """
    A comprehensive preprocessor for Vietnamese financial text.
    
    Attributes:
        special_tokens (List[str]): Special tokens to preserve
        stock_words (List[str]): List of stock codes to remove
    """
    
    def __init__(self):
        self.special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[MASK]']
        self.stock_words: List[str] = []
        self._dicchar = self._load_dicchar()
    
    def _load_dicchar(self) -> Dict[str, str]:
        """Load dictionary for Unicode normalization."""
        uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôốồổỗộơờớởỡợùúũủụưừứữửựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỆỄỂĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
        unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYY"
        
        dic = {}
        char1252 = "à|á|ả|ã|ạ|â|ầ|ấ|ẩ|ẫ|ậ|ă|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ê|ề|ế|ể|ễ|ệ|đ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ô|ố|ồ|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ù|ú|ũ|ủ|ụ|ư|ừ|ứ|ữ|ử|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Â|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ă|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ê|Ề|Ế|Ệ|Ễ|Ể|Đ|Í|Ì|Ỉ|Ĩ|Ị|Ó|Ò|Ỏ|Õ|Ọ|Ô|Ố|Ồ|Ổ|Ỗ|Ộ|Ơ|Ớ|Ờ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ư|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split("|")
        charutf8 = "à|á|ả|ã|ạ|â|ầ|ấ|ẩ|ẫ|ậ|ă|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ê|ề|ế|ể|ễ|ệ|đ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ô|ố|ồ|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ù|ú|ũ|ủ|ụ|ư|ừ|ứ|ữ|ử|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Â|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ă|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ê|Ề|Ế|Ệ|Ễ|Ể|Đ|Í|Ì|Ỉ|Ĩ|Ị|Ó|Ò|Ỏ|Õ|Ọ|Ô|Ố|Ồ|Ổ|Ỗ|Ộ|Ơ|Ớ|Ờ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ư|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split("|")
        
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic
    
    def convert_unicode(self, txt: str) -> str:
        """
        Normalize Unicode characters in Vietnamese text.
        
        Args:
            txt: Input text string
            
        Returns:
            Normalized text string
        """
        return regex.sub(
            r'à|á|ả|ã|ạ|â|ầ|ấ|ẩ|ẫ|ậ|ă|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ê|ề|ế|ể|ễ|ệ|đ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ô|ố|ồ|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ù|ú|ũ|ủ|ụ|ư|ừ|ứ|ữ|ử|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Â|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ă|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ê|Ề|Ế|Ệ|Ễ|Ể|Đ|Í|Ì|Ỉ|Ĩ|Ị|Ó|Ò|Ỏ|Õ|Ọ|Ô|Ố|Ồ|Ổ|Ỗ|Ộ|Ơ|Ớ|Ờ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ư|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: self._dicchar[x.group()], 
            txt
        )
    
    def process_special_words(self, text: str) -> str:
        """
        Handle special words like negation markers.
        
        Args:
            text: Input text
            
        Returns:
            Text with special words processed
        """
        new_text = ''
        text_lst = text.split()
        i = 0
        
        if 'không' in text_lst:
            while i <= len(text_lst) - 1:
                word = text_lst[i]
                if word == 'không':
                    next_idx = i + 1
                    if next_idx <= len(text_lst) - 1:
                        word = word + '_' + text_lst[next_idx]
                        i += 1
                new_text += word + ' '
                i += 1
        else:
            new_text = text
        
        return new_text.strip()
    
    def process_pos_tagging(self, text: str) -> str:
        """
        Keep only verbs, adjectives, and nouns; remove other parts of speech.
        
        Args:
            text: Input text
            
        Returns:
            Filtered text with only relevant POS tags
        """
        text = text + " a"
        new_document = ''
        
        for sentence in sent_tokenize(text):
            sentence = sentence.replace('.', '').lower()
            lst_word_type = ['V', 'N', 'A']
            
            try:
                tagged = pos_tag(sentence)
                sentence_words = ' '.join(
                    word[0].lower() if word[1].upper() in lst_word_type else ''
                    for word in tagged
                )
                new_document += sentence_words + ' '
            except Exception:
                new_document += sentence + ' '
        
        return new_document.strip()
    
    def remove_stock_codes(self, text: str, stock_words: Optional[List[str]] = None) -> str:
        """
        Remove stock codes from text.
        
        Args:
            text: Input text
            stock_words: List of stock codes to remove
            
        Returns:
            Text with stock codes removed
        """
        if stock_words is None:
            stock_words = self.stock_words
        
        document = ' '.join(
            '' if word.upper() in stock_words else word 
            for word in text.split()
        )
        return regex.sub(r'\s+', ' ', document).strip()
    
    def remove_time_expressions(self, text: str) -> str:
        """
        Remove time expressions (dates, times) from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with time expressions removed
        """
        # Remove dates with slashes
        document = ' '.join('' if word.find('/') != -1 else word for word in text.split())
        document = regex.sub(r'\s+', ' ', document).strip()
        
        # Remove time words
        time_words = ['ngày', 'tháng', 'năm', 'quý', 'lần']
        document = ' '.join('' if word in time_words else word for word in document.split())
        document = regex.sub(r'\s+', ' ', document).strip()
        
        return document
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove words containing numbers.
        
        Args:
            text: Input text
            
        Returns:
            Text with numeric words removed
        """
        document = ' '.join(
            '' if any(x.isdigit() for x in word) else word 
            for word in text.split()
        )
        return regex.sub(r'\s+', ' ', document).strip()
    
    def clean_text(self, text: str, stock_words: Optional[List[str]] = None) -> str:
        """
        Complete text cleaning pipeline.
        
        Args:
            text: Raw input text
            stock_words: Optional list of stock codes to remove
            
        Returns:
            Fully cleaned and processed text
        """
        document = text
        
        # Apply POS tagging and filtering
        document = self.process_pos_tagging(document)
        
        # Remove stock codes
        document = self.remove_stock_codes(document, stock_words)
        
        # Remove numbers
        document = self.remove_numbers(document)
        
        # Remove time expressions
        document = self.remove_time_expressions(document)
        
        return document
    
    def preprocess(self, text: str, stock_words: Optional[List[str]] = None) -> List[str]:
        """
        Full preprocessing pipeline returning tokenized text.
        
        Args:
            text: Raw input text
            stock_words: Optional list of stock codes to remove
            
        Returns:
            List of processed tokens
        """
        cleaned_text = self.clean_text(text, stock_words)
        tokens = word_tokenize(cleaned_text)
        return tokens
    
    def set_stock_words(self, stock_words: List[str]) -> None:
        """
        Set the list of stock codes to be removed during preprocessing.
        
        Args:
            stock_words: List of stock codes
        """
        self.stock_words = stock_words
