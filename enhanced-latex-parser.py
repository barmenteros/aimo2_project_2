"""
Enhanced LaTeX Parser - Core Components
Handles parsing of LaTeX mathematical notation for the AIMO competition
"""

import re
import logging
import sympy as sp
from sympy import parsing, latex
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from time import perf_counter
from contextlib import contextmanager

@dataclass
class ParsedComponent:
    """Represents a parsed LaTeX component with type information"""
    type: str
    original: str
    processed: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class LaTeXValidator:
    """Handles validation of LaTeX input and output"""
    
    @staticmethod
    def validate_balanced_delimiters(text: str) -> bool:
        """Check if delimiters are properly balanced"""
        stack = []
        delimiters = {
            '{': '}',
            '[': ']',
            '(': ')',
            r'\{': r'\}',
            r'\[': r'\]'
        }
        
        for i, char in enumerate(text):
            if char in '{[(':
                stack.append(char)
            elif char in '}])':
                if not stack:
                    return False
                if delimiters[stack[-1]] != char:
                    return False
                stack.pop()
                
        return len(stack) == 0

    @staticmethod
    def validate_math_delimiters(text: str) -> bool:
        """Check if math mode delimiters are properly paired"""
        count_inline = text.count('$') % 2 == 0
        count_display = text.count(r'\[') == text.count(r'\]')
        return count_inline and count_display

    @staticmethod
    def validate_commands(text: str) -> bool:
        """Validate LaTeX commands are properly formed"""
        # Check for unclosed commands
        if re.search(r'\\[a-zA-Z]+\s*\{[^\}]*$', text):
            return False
        return True

    @staticmethod
    def validate_expression(expression: str) -> bool:
        """
        Validate a mathematical expression
        
        Args:
            expression: Mathematical expression to validate
            
        Returns:
            True if expression is valid, False otherwise
        """
        try:
            # Check basic structure
            if not expression:
                return False
                
            # Check balanced parentheses and braces
            stack = []
            brackets = {'{': '}', '[': ']', '(': ')'}
            
            for char in expression:
                if char in brackets:
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return False
                    if char != brackets[stack.pop()]:
                        return False
            
            return len(stack) == 0
            
        except Exception:
            return False

    @staticmethod
    def sanitize_expression(expression: str) -> str:
        """
        Sanitize a mathematical expression
        
        Args:
            expression: Expression to sanitize
            
        Returns:
            Sanitized expression
        """
        try:
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[^a-zA-Z0-9\s\+\-\*\/\(\)\[\]\{\}\.,_\^\\]', '', expression)
            return sanitized
            
        except Exception:
            return ''
        
    @staticmethod
    def validate_competition_input(text: str) -> bool:
        """Validate input against competition requirements"""
        # Check length limits
        if len(text) > 10000:  # Arbitrary limit for safety
            return False
            
        # Check for unsupported environments
        unsupported = [
            r'\\begin{align',
            r'\\begin{gather',
            r'\\begin{multline',
            r'\\begin{array'
        ]
        for env in unsupported:
            if re.search(env, text):
                return False
                
        # Check for potentially problematic commands
        dangerous = [
            r'\\input',
            r'\\include',
            r'\\write',
            r'\\special'
        ]
        for cmd in dangerous:
            if re.search(cmd, text):
                return False
                
        return True

class LaTeXPattern:
    """Contains comprehensive LaTeX pattern definitions"""
    
    def __init__(self):
        """Initialize all pattern definitions"""
        # Math modes
        self.math_modes = {
            'inline_math': r'\$([^\$]+)\$',
            'display_math': r'\\\[([^\]]+)\\\]|\\\((.*?)\\\)',
            'equation': r'\\begin\{equation\}(.*?)\\end\{equation\}'
        }
        
        # Numbers and basic operations
        self.numbers = {
            'integer': r'(?<![a-zA-Z])-?\d+(?![a-zA-Z])',
            'decimal': r'(?<![a-zA-Z])-?\d+\.\d+(?![a-zA-Z])',
            'scientific': r'(?<![a-zA-Z])-?\d+(?:\.\d+)?[eE][+-]?\d+(?![a-zA-Z])'
        }
        
        # Operations
        self.operations = {
            'factorial': r'(\d+)!',
            'modulo': r'\\bmod|\\pmod{\w+}|\s+mod\s+',
            'fraction': r'\\frac\{([^\}]*)\}\{([^\}]*)\}',
            'binomial': r'\\binom\{([^\}]*)\}\{([^\}]*)\}'
        }
        
        # Functions
        self.functions = {
            'standard': r'\\(?:sin|cos|tan|log|ln|exp)(?:\s+|\{)',
            'floor_ceil': r'\\lfloor\s*(.*?)\s*\\rfloor|\\lceil\s*(.*?)\\rceil',
            'roots': r'\\sqrt(?:\[\d+\])?\{([^\}]+)\}'
        }
        
        # Sets and logic
        self.sets = {
            'set': r'\\{([^\\}]+)\\}',
            'set_ops': r'\\(?:cap|cup|setminus|subset|subseteq)',
            'number_sets': r'\\mathbb\{([NZQR])\}'
        }
        
        # Geometry
        self.geometry = {
            'angle': r'\\angle\s*([A-Z]{3})',
            'triangle': r'\\triangle\s*([A-Z]{3})',
            'degree': r'(\d+)\\degree|(\d+)Â°',
            'line': r'\\overline\{([A-Z]{2})\}'
        }
        
        # Subscripts and superscripts
        self.scripts = {
            'subscript': r'_\{([^\}]+)\}|_([a-zA-Z0-9])',
            'superscript': r'\^\{([^\}]+)\}|\^([a-zA-Z0-9])'
        }
        
        # Series and sequences
        self.sequences = {
            'dots': r'\\(?:cdots|ldots|dots)',
            'sum': r'\\sum_\{([^\}]*)\}\^\{([^\}]*)\}',
            'prod': r'\\prod_\{([^\}]*)\}\^\{([^\}]*)\}'
        }
        
        # Special characters
        self.special = {
            'greek': r'\\(?:alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma)',
            'operators': r'\\(?:times|div|pm|mp|cdot|circ)',
            'relations': r'\\(?:leq|geq|neq|approx|equiv|sim|in)',
            'symbols': r'\\(?:[,;:!]|\{|\}|\[|\]|\(|\))'
        }

        # Absolute value and intervals
        self.advanced = {
            'abs_value': r'\|([^|]+)\||\\left\|([^|]+)\\right\|',
            'interval_open': r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
            'interval_closed': r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',
            'interval_mixed': r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)|\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        }
        
        # Validate all patterns
        self._validate_patterns()
        
    def _validate_patterns(self):
        """Validate all regex patterns are properly formed"""
        for category in vars(self):
            if not category.startswith('_'):
                patterns = getattr(self, category)
                if isinstance(patterns, dict):
                    for name, pattern in patterns.items():
                        try:
                            re.compile(pattern)
                        except re.error as e:
                            raise ValueError(f"Invalid regex pattern in {category}.{name}: {str(e)}")

class EnhancedLaTeXParser:
    """Enhanced LaTeX parser with comprehensive notation support"""
    
    def __init__(self):
        """Initialize parser with all components"""
        try:
            self._setup_logger()
            self.patterns = LaTeXPattern()
            self.validator = LaTeXValidator()
            self.sympy_converter = SymPyConverter()
            self._initialize_handlers()
            self.component_registry = set()            
            
            # Validate initialization
            if not all([self.patterns, self.validator, self.handlers]):
                raise RuntimeError("Parser initialization failed")
                
            # Log successful initialization
            self.logger.info("LaTeX parser initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Parser initialization failed: {str(e)}")
        
        self.performance_metrics = defaultdict(list)
        
    processing_order = [
        'math_modes',
        'numbers',
        'operations',
        'functions',
        'sets',
        'geometry',
        'scripts',
        'sequences',
        'special',
        'advanced'
    ]
    
    @contextmanager
    def _track_time(self, operation: str):
        """Track execution time of operations using context manager"""
        start = perf_counter()
        try:
            yield
        finally:
            duration = perf_counter() - start
            self.performance_metrics[operation].append(duration)
            if duration > 25:  # Warning if operation takes >25s
                self.logger.warning(f"{operation} took {duration:.2f}s")

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations"""
        stats = {}
        for op, times in self.performance_metrics.items():
            stats[op] = {
                'avg': sum(times) / len(times),
                'max': max(times),
                'min': min(times),
                'total': sum(times)
            }
        return stats
    
    def _setup_logger(self):
        """Setup dedicated parser logger"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _initialize_handlers(self):
        """Initialize pattern handlers"""
        self.handlers = {
            # Math modes
            'inline_math': LaTeXHandlers.handle_inline_math,
            'display_math': LaTeXHandlers.handle_display_math,
            'equation': LaTeXHandlers.handle_display_math,
            
            # Numbers
            'integer': LaTeXHandlers.handle_integer,
            'decimal': LaTeXHandlers.handle_decimal,
            'scientific': LaTeXHandlers.handle_scientific,
            
            # Operations
            'factorial': LaTeXHandlers.handle_factorial,
            'modulo': LaTeXHandlers.handle_modulo,
            'fraction': LaTeXHandlers.handle_fraction,
            'binomial': LaTeXHandlers.handle_binomial,
            
            # Functions
            'standard': LaTeXHandlers.handle_standard_function,
            'floor_ceil': LaTeXHandlers.handle_floor_ceil,
            'roots': LaTeXHandlers.handle_roots,
            
            # Sets and logic
            'set': LaTeXHandlers.handle_set,
            'set_ops': LaTeXHandlers.handle_set_ops,
            'number_sets': LaTeXHandlers.handle_number_sets,
            
            # Geometry
            'angle': LaTeXHandlers.handle_angle,
            'triangle': LaTeXHandlers.handle_triangle,
            'degree': LaTeXHandlers.handle_degree,
            'line': LaTeXHandlers.handle_line,
            
            # Scripts
            'subscript': LaTeXHandlers.handle_subscript,
            'superscript': LaTeXHandlers.handle_superscript,
            
            # Sequences
            'dots': LaTeXHandlers.handle_dots,
            'sum': LaTeXHandlers.handle_sum,
            'prod': LaTeXHandlers.handle_prod,
            
            # Special characters
            'greek': LaTeXHandlers.handle_greek,
            'operators': LaTeXHandlers.handle_operators,
            'relations': LaTeXHandlers.handle_relations,
            
            # Advanced notation
            'abs_value': LaTeXHandlers.handle_abs_value,
            'interval_open': LaTeXHandlers.handle_interval_open,
            'interval_closed': LaTeXHandlers.handle_interval_closed,
            'interval_mixed': LaTeXHandlers.handle_interval_mixed
        }
        # Handlers will be implemented in the Handlers class
        
    def parse(self, latex_text: str) -> Dict[str, Any]:
        """Main parsing method for LaTeX text"""
        with self._track_time("total_parse"):
            try:
                self._cleanup_resources()  # Clean up before processing
                
                with self._track_time("input_validation"):
                    if not self._validate_input(latex_text):
                        raise ValueError("Invalid LaTeX input")
                
                with self._track_time("preprocessing"):
                    cleaned_text = self._preprocess_text(latex_text)
                
                # Process components
                components = []
                processed_text = cleaned_text
                
                # Process in specific order
                with self._track_time("pattern_processing"):
                    for category in self.processing_order:
                        patterns = getattr(self.patterns, category)
                        for pattern_name, pattern in patterns.items():
                            processed_text, new_components = self._process_pattern(
                                processed_text, pattern, pattern_name
                            )
                            components.extend(new_components)
                
                # Validate output
                with self._track_time("output_validation"):
                    if not self._validate_output(processed_text):
                        self.logger.warning("Output validation warnings present")
                
                self._cleanup_resources()  # Clean up after processing
                
                return {
                    'processed_text': processed_text,
                    'components': components,
                    'original_text': latex_text,
                    'success': True
                }
                
            except Exception as e:
                self.logger.error(f"Parsing error: {str(e)}")
                return {
                    'processed_text': latex_text,
                    'components': [],
                    'success': False,
                    'error': str(e)
                }
    
    def _validate_input(self, text: str) -> bool:
        """Validate input LaTeX text"""
        if not text or not isinstance(text, str):
            return False
            
        # Competition requirements check
        if not self.validator.validate_competition_input(text):
            self.logger.warning("Input fails competition requirements")
            return False
            
        # Structure validation - make this less strict
        if not self.validator.validate_balanced_delimiters(text):
            self.logger.warning("Unbalanced delimiters detected - proceeding with caution")
            # Don't return False here, just warn
            
        if not self.validator.validate_math_delimiters(text):
            self.logger.warning("Unmatched math delimiters detected - proceeding with caution")
            # Don't return False here, just warn
            
        if not self.validator.validate_commands(text):
            self.logger.warning("Invalid LaTeX commands detected - proceeding with caution")
            # Don't return False here, just warn
            
        return True
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize LaTeX text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        # Normalize spaces around math operators
        text = re.sub(r'([=<>+\-*/])', r' \1 ', text)
        
        # Normalize dots
        text = text.replace('...', r'\ldots')
        
        return text
    
    def _process_pattern(
        self, 
        text: str, 
        pattern: str, 
        pattern_name: str
    ) -> Tuple[str, List[ParsedComponent]]:
        """Process a single pattern in the text"""
        components = []
        current_text = text
        
        try:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                component = self._create_component(match, pattern_name)
                if component:
                    components.append(component)
                    if component.processed != component.original:
                        current_text = current_text.replace(
                            component.original,
                            component.processed
                        )
        except Exception as e:
            self.logger.error(f"Error processing pattern {pattern_name}: {str(e)}")
            
        return current_text, components
    
    def _create_component(self, match: re.Match, pattern_name: str) -> Optional[ParsedComponent]:
        """Create a parsed component from a match"""
        try:
            handler = self.handlers.get(pattern_name)
            if not handler:
                return None
                
            original = match.group(0)
            processed = handler(match)
            
            # Register component type
            self.component_registry.add(pattern_name)
            
            return ParsedComponent(
                type=pattern_name,
                original=original,
                processed=processed,
                metadata={
                    'span': match.span(),
                    'groups': match.groups()
                }
            )
        except Exception as e:
            self.logger.error(
                f"Error creating component for {pattern_name}: {str(e)}"
            )
            return None
    
    def _validate_output(self, text: str) -> bool:
        """Validate processed output"""
        # Use more specific pattern for unprocessed commands
        unprocessed = re.finditer(r'\\[a-zA-Z]+(?![a-zA-Z])', text)
        unprocessed_cmds = list(unprocessed)
        
        if unprocessed_cmds:
            self.logger.warning(
                f"Unprocessed LaTeX commands remain: {[m.group(0) for m in unprocessed_cmds]}"
            )
            
        # Verify structure
        if not self.validator.validate_balanced_delimiters(text):
            self.logger.warning("Output has unbalanced delimiters")
            
        return len(unprocessed_cmds) == 0  # Return False if unprocessed commands remain

    def get_component_statistics(self) -> Dict[str, int]:
        """Get statistics about parsed components"""
        stats = defaultdict(int)
        for component_type in self.component_registry:
            stats[component_type] += 1
        return dict(stats)
    
    def _cleanup_resources(self):
        """Clean up resources after processing"""
        import gc
        gc.collect()
        self.component_registry.clear()
    
    def debug_parse(self, latex_text: str) -> None:
        """Debug helper to show detailed parsing information"""
        result = self.parse(latex_text)
        if result['success']:
            print("Successfully parsed LaTeX text")
            print("\nComponents found:")
            for comp in result['components']:
                print(f"\nType: {comp.type}")
                print(f"Original: {comp.original}")
                print(f"Processed: {comp.processed}")
            print("\nProcessed text:", result['processed_text'])
            print("\nComponent statistics:")
            for comp_type, count in self.get_component_statistics().items():
                print(f"{comp_type}: {count}")
        else:
            print("Parsing failed:", result.get('error', 'Unknown error'))
            
    def get_parsing_trace(self, latex_text: str) -> Dict[str, Any]:
        """Get detailed trace of parsing process for debugging"""
        trace = {
            'preprocessing': None,
            'components_found': [],
            'error_locations': [],
            'unhandled_commands': []
        }
        
        try:
            # Track preprocessing
            trace['preprocessing'] = self._preprocess_text(latex_text)
            
            # Find unhandled LaTeX commands
            unhandled = re.finditer(r'\\[a-zA-Z]+', latex_text)
            trace['unhandled_commands'] = [
                (m.group(0), m.start()) for m in unhandled
            ]
            
            # Track component creation
            result = self.parse(latex_text)
            if result['success']:
                trace['components_found'] = [
                    (c.type, c.original) for c in result['components']
                ]
            
            return trace
            
        except Exception as e:
            trace['error'] = str(e)
            return trace
        
    def evaluate_math(self, latex_text: str, subs: Dict[str, Any] = None) -> Optional[Any]:
        """
        Enhanced evaluation of mathematical content in LaTeX text
        
        Args:
            latex_text: LaTeX text containing mathematical expressions
            subs: Dictionary of variable substitutions
            
        Returns:
            Evaluated result or None if evaluation fails
        """
        try:
            # First parse the LaTeX
            result = self.parse(latex_text)
            if not result['success']:
                return None
                
            # Extract all math mode content
            math_expressions = []
            
            # Handle display math mode
            display_matches = re.finditer(r'\\\[(.*?)\\\]', result['processed_text'])
            math_expressions.extend(m.group(1) for m in display_matches)
            
            # Handle inline math mode
            inline_matches = re.finditer(r'\$([^\$]+)\$', result['processed_text'])
            math_expressions.extend(m.group(1) for m in inline_matches)
            
            # Handle equation environment
            equation_matches = re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}', 
                                        result['processed_text'])
            math_expressions.extend(m.group(1) for m in equation_matches)
            
            if not math_expressions:
                # Try processing the entire text if no math mode found
                math_expressions = [result['processed_text']]
            
            # Process each expression
            results = []
            for expr in math_expressions:
                try:
                    # Convert to SymPy
                    sympy_expr = self.sympy_converter.latex_to_sympy(expr)
                    if sympy_expr is not None:
                        # Validate expression
                        if self.sympy_converter.validate_expression(sympy_expr):
                            # Evaluate with substitutions
                            result = self.sympy_converter.evaluate(sympy_expr, subs)
                            if result is not None:
                                results.append(float(result))
                except Exception as e:
                    logging.warning(f"Expression evaluation failed: {str(e)}")
                    continue
            
            # Return results
            if len(results) == 1:
                return results[0]
            elif len(results) > 1:
                return results
            return None
            
        except Exception as e:
            logging.error(f"Math evaluation error: {str(e)}")
            return None

class LaTeXHandlers:
    """Implements handlers for all LaTeX pattern types"""
    
    @staticmethod
    def handle_inline_math(match: re.Match) -> str:
        """Handle inline math mode content"""
        content = match.group(1)
        return content if content else ''

    @staticmethod
    def handle_display_math(match: re.Match) -> str:
        """Handle display math mode content"""
        content = match.group(1) or match.group(2)
        return content if content else ''

    @staticmethod
    def handle_integer(match: re.Match) -> str:
        """Handle integer numbers"""
        return match.group(0)

    @staticmethod
    def handle_decimal(match: re.Match) -> str:
        """Handle decimal numbers"""
        return match.group(0)

    @staticmethod
    def handle_scientific(match: re.Match) -> str:
        """Handle scientific notation"""
        return match.group(0)

    @staticmethod
    def handle_factorial(match: re.Match) -> str:
        """Handle factorial notation"""
        num = match.group(1)
        return f"factorial({num})"

    @staticmethod
    def handle_modulo(match: re.Match) -> str:
        """Handle modulo operations"""
        return " mod "

    @staticmethod
    def handle_fraction(match: re.Match) -> str:
        """Handle fraction notation"""
        num = match.group(1)
        den = match.group(2)
        return f"({num})/({den})"

    @staticmethod
    def handle_binomial(match: re.Match) -> str:
        """Handle binomial coefficients"""
        n = match.group(1)
        k = match.group(2)
        return f"C({n},{k})"

    @staticmethod
    def handle_standard_function(match: re.Match) -> str:
        """Handle standard mathematical functions"""
        return match.group(0).strip('\\{} ')

    @staticmethod
    def handle_floor_ceil(match: re.Match) -> str:
        """Handle floor and ceiling functions"""
        content = match.group(1) or match.group(2)
        if '\\lfloor' in match.group(0):
            return f"floor({content})"
        return f"ceil({content})"

    @staticmethod
    def handle_roots(match: re.Match) -> str:
        """Handle root expressions"""
        root_index = re.search(r'\[(\d+)\]', match.group(0))
        content = match.group(1)
        if root_index:
            return f"root({content},{root_index.group(1)})"
        return f"sqrt({content})"

    @staticmethod
    def handle_set(match: re.Match) -> str:
        """Handle set notation"""
        elements = match.group(1).split(',')
        return '{' + ', '.join(elem.strip() for elem in elements) + '}'

    @staticmethod
    def handle_set_ops(match: re.Match) -> str:
        """Handle set operations"""
        ops = {
            '\\cap': 'âˆ©',
            '\\cup': 'âˆª',
            '\\setminus': '\\',
            '\\subset': 'âŠ‚',
            '\\subseteq': 'âŠ†'
        }
        return ops.get(match.group(0), match.group(0))

    @staticmethod
    def handle_number_sets(match: re.Match) -> str:
        """Handle number set notation"""
        sets = {
            'N': 'â„•',  # Natural numbers
            'Z': 'â„¤',  # Integers
            'Q': 'â„š',  # Rational numbers
            'R': 'â„'   # Real numbers
        }
        return sets.get(match.group(1), match.group(0))

    @staticmethod
    def handle_angle(match: re.Match) -> str:
        """Handle angle notation"""
        vertices = match.group(1)
        return f"angle({vertices})"

    @staticmethod
    def handle_triangle(match: re.Match) -> str:
        """Handle triangle notation"""
        vertices = match.group(1)
        return f"triangle({vertices})"

    @staticmethod
    def handle_degree(match: re.Match) -> str:
        """Handle degree notation"""
        deg = match.group(1) or match.group(2)
        return f"{deg}Â°"

    @staticmethod
    def handle_line(match: re.Match) -> str:
        """Handle line segment notation"""
        points = match.group(1)
        return f"segment({points})"

    @staticmethod
    def handle_subscript(match: re.Match) -> str:
        """Handle subscript notation"""
        index = match.group(1) or match.group(2)
        return f"_{index}"

    @staticmethod
    def handle_superscript(match: re.Match) -> str:
        """Handle superscript notation"""
        power = match.group(1) or match.group(2)
        return f"^{power}"

    @staticmethod
    def handle_dots(match: re.Match) -> str:
        """Handle ellipsis notation"""
        return "..."

    @staticmethod
    def handle_sum(match: re.Match) -> str:
        """Handle summation notation"""
        lower = match.group(1)
        upper = match.group(2)
        return f"sum({lower},{upper})"

    @staticmethod
    def handle_prod(match: re.Match) -> str:
        """Handle product notation"""
        lower = match.group(1)
        upper = match.group(2)
        return f"prod({lower},{upper})"

    @staticmethod
    def handle_greek(match: re.Match) -> str:
        """Handle Greek letters"""
        greek = {
            'alpha': 'Î±',
            'beta': 'Î²',
            'gamma': 'Î³',
            'delta': 'Î´',
            'theta': 'Î¸',
            'lambda': 'Î»',
            'mu': 'Î¼',
            'pi': 'Ï€',
            'sigma': 'Ïƒ'
        }
        letter = match.group(0).strip('\\')
        return greek.get(letter, match.group(0))

    @staticmethod
    def handle_operators(match: re.Match) -> str:
        """Handle mathematical operators"""
        operators = {
            '\\times': 'Ã—',
            '\\div': 'Ã·',
            '\\pm': 'Â±',
            '\\mp': 'âˆ“',
            '\\cdot': 'Â·',
            '\\circ': 'Â°',
            '\\in': 'âˆˆ'
        }
        return operators.get(match.group(0), match.group(0))

    @staticmethod
    def handle_relations(match: re.Match) -> str:
        """Handle mathematical relations"""
        relations = {
            '\\leq': 'â‰¤',
            '\\geq': 'â‰¥',
            '\\neq': 'â‰ ',
            '\\approx': 'â‰ˆ',
            '\\equiv': 'â‰¡',
            '\\sim': 'âˆ¼',
            '\\in': 'âˆˆ'  # Add \in to relations map
        }
        return relations.get(match.group(0), match.group(0))
    
    @staticmethod
    def handle_abs_value(match: re.Match) -> str:
        """Handle absolute value notation"""
        content = match.group(1) or match.group(2)
        return f"abs({content})"
    
    @staticmethod
    def handle_interval_open(match: re.Match) -> str:
        """Handle open interval notation"""
        start, end = match.group(1), match.group(2)
        return f"interval_open({start},{end})"
    
    @staticmethod
    def handle_interval_closed(match: re.Match) -> str:
        """Handle closed interval notation"""
        start, end = match.group(1), match.group(2)
        return f"interval_closed({start},{end})"
    
    @staticmethod
    def handle_interval_mixed(match: re.Match) -> str:
        """Handle mixed interval notation"""
        if match.group(1) and match.group(2):  # [a,b)
            return f"interval_left_closed({match.group(1)},{match.group(2)})"
        else:  # (a,b]
            return f"interval_right_closed({match.group(3)},{match.group(4)})"
        
    @staticmethod
    def handle_symbols(match: re.Match) -> str:
        """Handle escaped symbols"""
        symbol_map = {
            '\\{': '{',
            '\\}': '}',
            '\\[': '[',
            '\\]': ']',
            '\\(': '(',
            '\\)': ')',
            '\\,': ' ',
            '\\;': ' ',
            '\\:': ' ',
            '\\!': ''
        }
        return symbol_map.get(match.group(0), match.group(0))

class LaTeXProcessor:
    """Processes complete LaTeX expressions using the handlers"""
    
    def __init__(self):
        """Initialize processor with handlers"""
        self.handlers = LaTeXHandlers()
        self.logger = logging.getLogger(__name__)

    def process_math_content(self, content: str) -> str:
        """
        Process mathematical content within delimiters
        
        Args:
            content: Mathematical content to process
            
        Returns:
            Processed mathematical content
        """
        if not content:
            return ''
            
        try:
            processed = content
            
            # Process in specific order
            processing_steps = [
                # Handle symbols and operators first
                (r'\\(?:[,;:!]|\{|\}|\[|\]|\(|\))', self.handlers.handle_symbols),
                (r'\\(?:times|div|pm|mp|cdot|circ)', self.handlers.handle_operators),
                (r'\\(?:leq|geq|neq|approx|equiv|sim|in)', self.handlers.handle_relations),
                
                # Handle nested structures
                (r'\\frac\{([^\}]*)\}\{([^\}]*)\}', self.handlers.handle_fraction),
                (r'\\binom\{([^\}]*)\}\{([^\}]*)\}', self.handlers.handle_binomial),
                (r'\\sqrt(?:\[\d+\])?\{([^\}]+)\}', self.handlers.handle_roots),
                
                # Handle functions and values
                (r'\|([^|]+)\||\\left\|([^|]+)\\right\|', self.handlers.handle_abs_value),
                (r'\\(?:sin|cos|tan|log|ln|exp)(?:\s+|\{)', self.handlers.handle_standard_function),
                
                # Handle subscripts and superscripts
                (r'_\{([^\}]+)\}|_([a-zA-Z0-9])', self.handlers.handle_subscript),
                (r'\^\{([^\}]+)\}|\^([a-zA-Z0-9])', self.handlers.handle_superscript),
                
                # Handle special characters
                (r'\\(?:alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma)', self.handlers.handle_greek),
                
                # Handle intervals last
                (r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', self.handlers.handle_interval_open),
                (r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', self.handlers.handle_interval_closed),
                (r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)|\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', self.handlers.handle_interval_mixed)
            ]
            
            # Apply each processing step
            for pattern, handler in processing_steps:
                matches = list(re.finditer(pattern, processed))
                for match in reversed(matches):  # Process from right to left
                    replacement = handler(match)
                    processed = processed[:match.start()] + replacement + processed[match.end():]
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing math content: {str(e)}")
            return content

    def process_complete_expression(self, expression: str) -> str:
        """
        Process a complete mathematical expression
        
        Args:
            expression: Complete LaTeX expression
            
        Returns:
            Fully processed expression
        """
        try:
            # First pass: Process math mode content
            math_pattern = r'\$([^\$]+)\$|\\\[(.*?)\\\]|\\\((.*?)\\\)'
            matches = list(re.finditer(math_pattern, expression))
            
            processed = expression
            for match in reversed(matches):
                content = match.group(1) or match.group(2) or match.group(3)
                processed_content = self.process_math_content(content)
                processed = processed[:match.start()] + processed_content + processed[match.end():]
            
            # Second pass: Process remaining LaTeX commands
            processed = self._process_remaining_commands(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing complete expression: {str(e)}")
            return expression

    def _process_remaining_commands(self, text: str) -> str:
        """Process any remaining LaTeX commands"""
        try:
            # Handle any remaining special cases
            replacements = [
                (r'\\text\{([^\}]+)\}', r'\1'),
                (r'\\left', ''),
                (r'\\right', ''),
                (r'\\,', ' '),
                (r'\\;', ' '),
                (r'\\:', ' '),
                (r'\\!', '')
            ]
            
            processed = text
            for pattern, replacement in replacements:
                processed = re.sub(pattern, replacement, processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing remaining commands: {str(e)}")
            return text

class SymPyConverter:
    """Handles conversion between LaTeX and SymPy expressions"""
    
    def __init__(self):
        """Initialize with correct transformations and symbols"""
        # Set up transformations for parsing
        self.transformations = (
            parsing.sympy_parser.standard_transformations + 
            (parsing.sympy_parser.implicit_multiplication,
             parsing.sympy_parser.implicit_application,
             parsing.sympy_parser.convert_xor)
        )
        
        # Initialize parser with transformations
        self.local_dict = {}
        self.global_dict = {
            "Symbol": sp.Symbol,
            "Integer": sp.Integer,
            "Float": sp.Float,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "log": sp.log,
            "exp": sp.exp,
            "pi": sp.pi,
            "E": sp.E
        }
        
        self._setup_symbols()
        
    def _setup_symbols(self):
        """Initialize common symbols and add to local dict"""
        # Define basic variables
        symbols = 'x y z n m k a b c'
        greek = 'alpha beta gamma delta theta lambda mu pi sigma'
        
        # Create and store all symbols
        for sym in (symbols + ' ' + greek).split():
            self.local_dict[sym] = sp.Symbol(sym)
            if sym in ['alpha', 'beta']:  # Ensure Greek letters are properly defined
                self.global_dict[sym] = sp.Symbol(sym)
        
        # Store special constants and functions
        special_symbols = {
            'pi': sp.pi,
            'E': sp.E,
            'I': sp.I,
            'oo': sp.oo,
            'sqrt': sp.sqrt,
            'Eq': sp.Eq
        }
        
        self.local_dict.update(special_symbols)
        self.global_dict.update(special_symbols)
        
    def latex_to_sympy(self, latex_expr: str) -> Optional[sp.Expr]:
        """Convert LaTeX to SymPy with proper parsing"""
        try:
            print(f"\nDEBUG: Starting LaTeX to SymPy conversion")
            print(f"DEBUG: Input LaTeX: {latex_expr}")
            
            # Preprocess LaTeX
            expr_str = self._preprocess_latex(latex_expr)
            print(f"DEBUG: After preprocessing: {expr_str}")
            
            # Handle equations separately
            if '=' in expr_str:
                print("DEBUG: Equation detected, parsing left and right sides")
                left_str, right_str = expr_str.split('=')
                try:
                    left_expr = parsing.sympy_parser.parse_expr(
                        left_str.strip(),
                        local_dict=self.local_dict,
                        global_dict=self.global_dict,
                        transformations=self.transformations
                    )
                    right_expr = parsing.sympy_parser.parse_expr(
                        right_str.strip(),
                        local_dict=self.local_dict,
                        global_dict=self.global_dict,
                        transformations=self.transformations
                    )
                    result = sp.Eq(left_expr, right_expr)
                    print(f"DEBUG: Created equation: {result}")
                    return result
                except Exception as eq_error:
                    print(f"DEBUG: Equation parsing error: {str(eq_error)}")
                    return None
            
            # Handle regular expressions
            try:
                result = parsing.sympy_parser.parse_expr(
                    expr_str,
                    local_dict=self.local_dict,
                    global_dict=self.global_dict,
                    transformations=self.transformations,
                    evaluate=True
                )
                print(f"DEBUG: Parsed regular expression: {result}")
                return result
                
            except Exception as parse_error:
                print(f"DEBUG: Regular expression parsing failed: {str(parse_error)}")
                return None
                
        except Exception as e:
            print(f"DEBUG: LaTeX conversion failed: {str(e)}")
            return None
            
    def _preprocess_latex(self, latex_expr: str) -> str:
        """Enhanced preprocessing with debug messages"""
        try:
            print("\nDEBUG: Starting LaTeX preprocessing")
            expr = latex_expr
            
            # Handle fractions first
            print("DEBUG: Processing fractions")
            while '\\frac' in expr:
                match = re.search(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', expr)
                if not match:
                    break
                num, den = match.groups()
                replacement = f"({num})/({den})"
                expr = expr[:match.start()] + replacement + expr[match.end():]
                print(f"DEBUG: Fraction processed: {expr}")
            
            print("DEBUG: Processing special symbols")
            # Handle square roots
            expr = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', expr)
            print(f"DEBUG: After sqrt: {expr}")
            
            # Handle special symbols
            replacements = [
                ('\\alpha', 'alpha'),
                ('\\beta', 'beta'),
                ('\\pi', 'pi'),
                ('\\', '')  # Remove remaining backslashes
            ]
            
            for old, new in replacements:
                if old in expr:
                    expr = expr.replace(old, new)
                    print(f"DEBUG: Replaced {old} with {new}: {expr}")
            
            # Final cleanup
            expr = expr.strip()
            print(f"DEBUG: Final preprocessed expression: {expr}")
            
            return expr
            
        except Exception as e:
            print(f"DEBUG: Preprocessing error: {str(e)}")
            return latex_expr

    def _is_valid_sympy_expr(self, expr_str: str) -> bool:
        """Check if string can be parsed as valid SymPy expression"""
        try:
            expr = self.parser(expr_str)
            str(expr)  # Verify expression can be stringified
            return True
        except Exception:
            return False
    
    def validate_expression(self, expr: sp.Expr) -> bool:
        """Validate a SymPy expression"""
        try:
            # Check if expression can be evaluated
            expr.free_symbols  # Access free symbols to validate
            return True
        except Exception:
            return False

    def evaluate(self, expr: sp.Expr, subs: Dict[str, Any] = None) -> Optional[Any]:
        """Enhanced evaluation with debug messages"""
        try:
            print("\nDEBUG: Starting expression evaluation")
            print(f"DEBUG: Input expression: {expr}")
            print(f"DEBUG: Substitutions: {subs}")
            
            if expr is None:
                print("DEBUG: Expression is None")
                return None
                
            # Handle equations
            if isinstance(expr, sp.Eq):
                print("DEBUG: Processing equation")
                if subs:
                    expr = expr.subs(subs)
                    print(f"DEBUG: After substitution: {expr}")
                result = float((expr.lhs - expr.rhs).evalf())
                print(f"DEBUG: Equation evaluation result: {result}")
                return result
                
            # Handle regular expressions
            if subs:
                expr = expr.subs(subs)
                print(f"DEBUG: After substitution: {expr}")
                
            try:
                result = float(expr.evalf())
                print(f"DEBUG: Direct evaluation result: {result}")
                return result
            except (TypeError, ValueError) as e:
                print(f"DEBUG: Direct evaluation failed: {str(e)}")
                try:
                    result = float(sp.N(expr, 15))
                    print(f"DEBUG: N evaluation result: {result}")
                    return result
                except (TypeError, ValueError) as e:
                    print(f"DEBUG: N evaluation failed: {str(e)}")
                    print(f"DEBUG: Returning string representation")
                    return str(expr)
                    
        except Exception as e:
            print(f"DEBUG: Evaluation error: {str(e)}")
            return None