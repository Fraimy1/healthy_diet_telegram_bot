import re
import pandas as pd

class Parser:
    def __init__(self, verbose=False):
        """
        Initializes the Parser with precompiled regex patterns.

        Parameters:
        - verbose (bool): If True, prints additional debug information.
        """
        self.verbose = verbose

        # Define a number pattern that allows spaces between digits and decimal points
        self._number_pattern = r'-?\d+(?:\s*[.,]\s*\d+)?'

        # Update patterns to accommodate spaces in numbers and units
        self._hierarchical_number_pattern = re.compile(r'\b(\d+(?:[.\-]\d+)+)\b')

        self._percentage_pattern = re.compile(
            rf'({self._number_pattern}(?:\s*-\s*{self._number_pattern})?)\s*%', re.IGNORECASE
        )

        # Make 'units' an instance variable
        self._units = r'(?:г|гр|кг|мг|мкг|л|мл|см|ml|l|kg|g|gm|cm)\b'
        self._amount_pattern = re.compile(
            rf'({self._number_pattern})\s*{self._units}', re.IGNORECASE
        )

        self._portion_pattern = re.compile(
            rf'({self._number_pattern})\s*(?:шт|порц|упак|доз)\b', re.IGNORECASE
        )

        self._cost_pattern = re.compile(
            rf'({self._number_pattern})\s*(?:руб|р|₽)\b', re.IGNORECASE
        )

        self._code_pattern = re.compile(
            r'(по\s*(\d+(?:[.\-]\d+)*))', re.IGNORECASE
        )

        self._empty_parentheses_pattern = re.compile(r'\(\s*\)')

        if self.verbose:
            print("Parser initialized with updated regex patterns.")

    # === Pattern Matching Methods ===

    def extract_hierarchical_number(self, entry):
        """Extracts hierarchical number from the entry."""
        match = self._hierarchical_number_pattern.search(entry)
        if match:
            info = match.group(1)
            entry = entry.replace(match.group(0), '').strip()
            return info, entry
        return None, entry

    def extract_percentage(self, entry):
        """Extracts percentage from the entry."""
        match = self._percentage_pattern.search(entry)
        if match:
            # Remove spaces within the number and replace commas with dots
            info = match.group(1).replace(' ', '').replace(',', '.')
            entry = entry.replace(match.group(0), '').strip()
            return info + '%', entry
        return None, entry

    def extract_amount(self, entry):
        """Extracts amount with units from the entry."""
        match = self._amount_pattern.search(entry)
        if match:
            # Remove spaces within the number and replace commas with dots
            number = match.group(1).replace(' ', '').replace(',', '.')
            unit_match = re.search(rf'{self._units}', match.group(0), re.IGNORECASE)
            unit = unit_match.group(0) if unit_match else ''
            info = number + ' ' + unit.lower()
            info = self.replace_multiple_spaces_with_single(info)
            entry = entry.replace(match.group(0), '').strip()
            return info, entry
        return None, entry

    def extract_portion(self, entry):
        """Extracts portion information from the entry."""
        match = self._portion_pattern.search(entry)
        if match:
            # The first capturing group is the numeric part.
            number_str = match.group(1).replace(' ', '').replace(',', '.')

            # Now we want the portion unit from the entire matched string.
            # e.g., if match.group(0) == "3 шт" or "2 упак"
            portion_text = match.group(0)

            # We'll do a quick second regex to isolate the unit.
            portion_unit_pattern = re.compile(r'(шт|порц|упак|доз)', re.IGNORECASE)
            unit_match = portion_unit_pattern.search(portion_text)
            if unit_match:
                portion_unit = unit_match.group(1).lower()
            else:
                portion_unit = ""  # Fallback if it somehow didn't match

            info = f"{number_str} {portion_unit}"
            info = self.replace_multiple_spaces_with_single(info)
            
            # Remove the matched substring from the original entry
            entry = entry.replace(match.group(0), '').strip()
            return info, entry
        return None, entry

    def extract_cost(self, entry):
        """Extracts cost from the entry."""
        match = self._cost_pattern.search(entry)
        if match:
            info = match.group(0).replace(' ', '').replace(',', '.').lower()
            entry = entry.replace(match.group(0), '').strip()
            return info, entry
        return None, entry

    def extract_code(self, entry):
        """Extracts code from the entry."""
        match = self._code_pattern.search(entry)
        if match:
            info = match.group(1)
            entry = entry.replace(match.group(0), '').strip()
            return info, entry
        return None, entry

    # === Cleaning Methods ===

    def replace_double_periods(self, entry):
        """Replaces double periods with a single period."""
        return entry.replace('..', '.').strip()

    def remove_empty_parentheses(self, entry):
        """Removes empty parentheses from the entry."""
        return self._empty_parentheses_pattern.sub('', entry).strip()

    def replace_underscores_with_spaces(self, entry):
        """Replaces underscores with spaces in a data entry."""
        return re.sub(r'_+', ' ', entry)

    def strip_trailing_dots_and_spaces(self, entry):
        """Strips trailing dots and spaces from a data entry."""
        return entry.rstrip(' .')

    def convert_to_lowercase(self, entry):
        """Converts a data entry to lower case."""
        return entry.lower()

    def add_spaces_around_punctuation(self, entry):
        """Adds spaces before and after each punctuation character in a data entry."""
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        pattern = f"([{re.escape(punctuation)}])"
        return re.sub(pattern, r' \1 ', entry)

    def normalize_units_in_amount(self, amount):
        """Normalizes units in a single amount amount."""
        if amount is None:
            return None
        
        unit_replacements = {
            'kg': 'кг',
            'g': 'г',
            'гр': 'г',
            'mg': 'мг',
            'мг': 'мг',
            'мкг': 'мкг',
            'l': 'л',
            'ml': 'мл',
            'gm': 'г',
            "гм": "г",
            "мгм": "мг",
            "мкгм": "мкг",
            "лм": "л",
            "млм": "мл",
        }
        number, unit = amount.split()
        normalized_unit = unit_replacements.get(unit.lower(), unit)
        return f"{number} {normalized_unit}"

    def convert_amount_units_to_grams(self, amount):
        """Converts units in amount to grams and returns the value in grams for the entry."""
        unit_conversion = {
            'кг': 1000, 
            'г': 1,
            'мг': 0.001,
            'мкг': 0.000001,
            'л': 1000,
            'мл': 1,
        }

        number, unit = amount.split()
        multiplier = unit_conversion.get(unit.lower(), 1000)  # default to kg
        amount_in_grams = float(number) * multiplier
        
        return amount_in_grams

    def split_on_case_change(self, entry):
        """Splits concatenated words on case change."""
        return re.sub(r'(?<!^)(?<![А-Я])(?=[А-Я])', ' ', entry)

    def separate_numbers_and_letters(self, entry):
        """Inserts spaces between letters and numbers."""
        entry = re.sub(r'([а-яА-Яa-zA-Z])(\d)', r'\1 \2', entry)
        entry = re.sub(r'(\d)([а-яА-Яa-zA-Z])', r'\1 \2', entry)
        return entry

    def remove_numbers(self, entry):
        """Removes all numbers from the entry."""
        return re.sub(r'\d+', '', entry)

    def replace_multiple_spaces_with_single(self, entry):
        """Replaces multiple spaces with a single space."""
        return re.sub(r'\s+', ' ', entry).strip()

    def replace_punctuation_with_spaces(self, entry):
        """Replaces all punctuation characters with spaces."""
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        translator = str.maketrans({key: ' ' for key in punctuation})
        return entry.translate(translator)

    # === Core Methods ===

    def clean_entry(self, entry, **kwargs):
        """
        Cleans a single data entry.

        Parameters:
        - entry (str): The data entry to clean.
        - **kwargs: Cleaning options.

        Returns:
        - entry (str): Cleaned entry.
        """
        default_options = {
            'replace_double_periods': True,
            'remove_empty_parentheses': True,
            'replace_underscores_with_spaces': True,
            'split_on_case_change': True,
            'separate_numbers_and_letters': True,
            'normalize_units': True,
            'add_spaces_around_punctuation': True,
            'replace_punctuation_with_spaces': False,
            'remove_numbers': False,
            'strip_trailing_dots_and_spaces': True,
            'convert_to_lowercase': True,
            'replace_multiple_spaces_with_single': True,
        }
        options = {**default_options, **kwargs}

        for key in kwargs:
            if key not in default_options:
                raise ValueError(f"Unknown option '{key}' provided to clean_entry.")

        if options['replace_double_periods']:
            entry = self.replace_double_periods(entry)
        if options['remove_empty_parentheses']:
            entry = self.remove_empty_parentheses(entry)
        if options['replace_underscores_with_spaces']:
            entry = self.replace_underscores_with_spaces(entry)
        if options['split_on_case_change']:
            entry = self.split_on_case_change(entry)
        if options['separate_numbers_and_letters']:
            entry = self.separate_numbers_and_letters(entry)
        if options['add_spaces_around_punctuation']:
            entry = self.add_spaces_around_punctuation(entry)
        if options['replace_punctuation_with_spaces']:
            entry = self.replace_punctuation_with_spaces(entry)
        if options['remove_numbers']:
            entry = self.remove_numbers(entry)
        if options['strip_trailing_dots_and_spaces']:
            entry = self.strip_trailing_dots_and_spaces(entry)
        if options['convert_to_lowercase']:
            entry = self.convert_to_lowercase(entry)
        if options['replace_multiple_spaces_with_single']:
            entry = self.replace_multiple_spaces_with_single(entry)

        return entry

    def clean_dataset(self, data, **kwargs):
        """Cleans the dataset."""
        return [self.clean_entry(entry, **kwargs) for entry in data]

    def parse_entry(self, entry, preprocessing=True, **kwargs):
        """
        Parses a single data entry and extracts relevant information.

        Parameters:
        - entry (str): The data entry to parse.
        - preprocessing (bool): If True, perform initial cleaning steps before parsing.
        - **kwargs: Options for extraction, similar to clean_entry.

        Returns:
        - info (dict): A dictionary containing extracted information.
        """
        default_options = {
            'extract_hierarchical_number': True,
            'extract_percentage': True,
            'extract_amount': True,
            'extract_portion': True,
            'extract_cost': True,
            'extract_code': True,
        }

        for key in kwargs:
            if key not in default_options:
                raise ValueError(f"Unknown option '{key}' provided to parse_entry.")

        options = {**default_options, **kwargs}

        # Perform cleaning steps required for parsing
        if preprocessing:
            entry = self.clean_entry(
                entry,
                remove_empty_parentheses=False,
                normalize_units=False,
            )

        # Proceed with extraction
        info = {'original_entry': entry}

        if options.get('extract_hierarchical_number', True):
            info['hierarchical_number'], entry = self.extract_hierarchical_number(entry)
        if options.get('extract_percentage', True):
            info['percentage'], entry = self.extract_percentage(entry)
        if options.get('extract_amount', True):
            info['amount'], entry = self.extract_amount(entry)
            if info.get('amount'):
                info['amount'] = self.normalize_units_in_amount(info['amount'])
        if options.get('extract_portion', True):
            info['portion'], entry = self.extract_portion(entry)
        if options.get('extract_cost', True):
            info['cost'], entry = self.extract_cost(entry)
        if options.get('extract_code', True):
            info['code'], entry = self.extract_code(entry)

        entry = self.remove_empty_parentheses(entry)
        info['product_name'] = entry.strip()
        info = {k: v for k, v in info.items() if v is not None}

        if self.verbose:
            print(f"Parsed entry: {info}")

        return info

    def parse_dataset(self, data, **kwargs):
        """Parses the dataset and returns a pandas DataFrame."""
        default_options = {
            'extract_hierarchical_number': True,
            'extract_percentage': True,
            'extract_amount': True,
            'extract_portion': True,
            'extract_cost': True,
            'extract_code': True,
        }
        options = {**default_options, **kwargs}

        for key in kwargs:
            if key not in default_options:
                raise ValueError(f"Unknown option '{key}' provided to parse_dataset.")

        parsed_data = []
        for entry in data:
            info = self.parse_entry(entry, options)
            parsed_data.append(info)

        df = pd.DataFrame(parsed_data)
        return df

    def normalize_units(self, data):
        """Normalizes units in the data entries."""
        return [self.normalize_units_in_entry(entry) for entry in data]
