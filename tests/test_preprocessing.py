"""
Unit Tests for Vietnamese Text Preprocessor

This module contains unit tests for the text preprocessing functionality.
"""

import pytest
from src.data_processing.preprocessor import VietnameseTextPreprocessor


class TestVietnameseTextPreprocessor:
    """Test suite for VietnameseTextPreprocessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = VietnameseTextPreprocessor()
    
    def test_convert_unicode_basic(self):
        """Test basic Unicode normalization."""
        text = "Xin chào Việt Nam"
        result = self.preprocessor.convert_unicode(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_remove_numbers(self):
        """Test number removal from text."""
        text = "Giá cổ phiếu tăng 15% trong ngày 25/12"
        result = self.preprocessor.remove_numbers(text)
        assert "15" not in result
        assert "25/12" not in result
    
    def test_remove_stock_codes(self):
        """Test stock code removal."""
        text = "Cổ phiếu VIC và VNM đều tăng điểm"
        stock_words = ["VIC", "VNM"]
        result = self.preprocessor.remove_stock_codes(text, stock_words)
        assert "VIC" not in result.upper()
        assert "VNM" not in result.upper()
    
    def test_remove_time_expressions(self):
        """Test time expression removal."""
        text = "Ngày 15 tháng 12 năm 2023 quý 4"
        result = self.preprocessor.remove_time_expressions(text)
        assert "ngày" not in result.lower()
        assert "tháng" not in result.lower()
        assert "năm" not in result.lower()
        assert "quý" not in result.lower()
    
    def test_clean_text_pipeline(self):
        """Test complete text cleaning pipeline."""
        text = "VIC: Giá cổ phiếu tăng 15% vào ngày 25/12/2023"
        stock_words = ["VIC"]
        result = self.preprocessor.clean_text(text, stock_words)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_empty_input(self):
        """Test handling of empty input."""
        text = ""
        result = self.preprocessor.clean_text(text)
        assert result == ""
    
    def test_special_characters_removal(self):
        """Test special character handling."""
        text = "Cổ phiếu tăng mạnh!!! Giá: 50.000đ"
        result = self.preprocessor.clean_text(text)
        assert isinstance(result, str)
    
    def test_set_stock_words(self):
        """Test setting stock words list."""
        stock_words = ["VIC", "VNM", "HPG"]
        self.preprocessor.set_stock_words(stock_words)
        assert self.preprocessor.stock_words == stock_words


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = VietnameseTextPreprocessor()
    
    def test_none_input_handling(self):
        """Test that None input raises appropriate error."""
        with pytest.raises(AttributeError):
            self.preprocessor.clean_text(None)
    
    def test_very_long_text(self):
        """Test processing of very long text."""
        text = "Tăng giá. " * 1000
        result = self.preprocessor.clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mixed_languages(self):
        """Test text with mixed Vietnamese and English."""
        text = "Stock VIC increases mạnh trong phiên giao dịch"
        result = self.preprocessor.clean_text(text)
        assert isinstance(result, str)
    
    def test_only_numbers(self):
        """Test text containing only numbers."""
        text = "123 456 789"
        result = self.preprocessor.remove_numbers(text)
        assert result.strip() == ""
    
    def test_only_stock_codes(self):
        """Test text containing only stock codes."""
        text = "VIC VNM HPG"
        stock_words = ["VIC", "VNM", "HPG"]
        result = self.preprocessor.remove_stock_codes(text, stock_words)
        assert result.strip() == ""


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
