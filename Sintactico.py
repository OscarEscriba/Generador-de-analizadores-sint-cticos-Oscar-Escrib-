
import sys
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class Production:
    """Representa una producción de la gramática"""
    left: str  # No-terminal del lado izquierdo
    right: List[str]  # Símbolos del lado derecho
    
    def __str__(self):
        return f"{self.left} -> {' '.join(self.right) if self.right else 'ε'}"

@dataclass
class Item:
    """Representa un item LR(0)"""
    production: Production
    dot_position: int
    
    def __hash__(self):
        return hash((self.production.left, tuple(self.production.right), self.dot_position))
    
    def __eq__(self, other):
        return (isinstance(other, Item) and 
                self.production.left == other.production.left and
                self.production.right == other.production.right and
                self.dot_position == other.dot_position)
    
    def __str__(self):
        right = self.production.right[:]
        right.insert(self.dot_position, '•')
        return f"{self.production.left} -> {' '.join(right)}"
    
    def is_complete(self) -> bool:
        """Retorna True si el punto está al final"""
        return self.dot_position >= len(self.production.right)
    
    def next_symbol(self) -> Optional[str]:
        """Retorna el símbolo después del punto, o None si está al final"""
        if self.is_complete():
            return None
        return self.production.right[self.dot_position]
    
    def advance(self):
        """Retorna un nuevo item con el punto avanzado"""
        return Item(self.production, self.dot_position + 1)

class YALexProcessor:
    """Procesador de archivos YALex para generar tokens"""
    
    def __init__(self):
        self.tokens = []
        
    def process_yalex_file(self, filename: str) -> List[str]:
        """Procesa un archivo .yalex y extrae los tokens definidos"""
        if not os.path.exists(filename):
            print(f"Advertencia: Archivo YALex '{filename}' no encontrado")
            return []
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Buscar definiciones de tokens en formato YALex
            # Formato típico: token_name = "pattern"
            tokens = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if '=' in line and not line.startswith('//') and not line.startswith('/*'):
                    # Extraer nombre del token
                    token_name = line.split('=')[0].strip()
                    if token_name and token_name.isupper():
                        tokens.append(token_name)
            
            return tokens
        except Exception as e:
            print(f"Error procesando archivo YALex: {e}")
            return []
    """Parser para archivos .yalp"""
    
    def __init__(self):
        self.tokens = []
        self.ignore_tokens = set()
        self.productions = []
        
    def parse_file(self, filename: str):
        """Parsea un archivo .yalp"""
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remover comentarios
        content = self._remove_comments(content)
        
        # Dividir en secciones - buscar %% como línea separada
        lines = content.split('\n')
        separator_found = False
        separator_index = -1
        
        for i, line in enumerate(lines):
            if line.strip() == '%%':
                if separator_found:
                    raise ValueError("El archivo tiene múltiples líneas %% - debe tener exactamente una")
                separator_found = True
                separator_index = i
        
        if not separator_found:
            raise ValueError("El archivo debe tener exactamente una línea %% separando tokens y producciones")
        
        tokens_section = '\n'.join(lines[:separator_index])
        productions_section = '\n'.join(lines[separator_index + 1:])
        
        tokens_section, productions_section = tokens_section, productions_section
        
        # Parsear tokens
        self._parse_tokens(tokens_section.strip())
        
        # Parsear producciones
        self._parse_productions(productions_section.strip())
        
        return self.tokens, self.ignore_tokens, self.productions
    
    def _remove_comments(self, content: str) -> str:
        """Remueve comentarios /* ... */"""
        result = []
        i = 0
        while i < len(content):
            if i < len(content) - 1 and content[i:i+2] == '/*':
                # Encontrar el cierre del comentario
                j = i + 2
                while j < len(content) - 1:
                    if content[j:j+2] == '*/':
                        i = j + 2
                        break
                    j += 1
                else:
                    raise ValueError("Comentario no cerrado")
            else:
                result.append(content[i])
                i += 1
        return ''.join(result)
    
    def _parse_tokens(self, tokens_section: str):
        """Parsea la sección de tokens"""
        lines = [line.strip() for line in tokens_section.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith('%token'):
                # Extraer tokens
                token_part = line[6:].strip()  # Remover '%token'
                tokens = token_part.split()
                self.tokens.extend(tokens)
            elif line.startswith('IGNORE'):
                # Extraer tokens a ignorar
                ignore_part = line[6:].strip()  # Remover 'IGNORE'
                ignore_tokens = ignore_part.split()
                self.ignore_tokens.update(ignore_tokens)
    
    def _parse_productions(self, productions_section: str):
        """Parsea la sección de producciones"""
        # Dividir por producciones (terminadas en ;)
        productions_text = productions_section.replace('\n', ' ')
        
        # Encontrar todas las producciones
        current_production = ""
        for char in productions_text:
            current_production += char
            if char == ';':
                self._parse_single_production(current_production.strip())
                current_production = ""
    
    def _parse_single_production(self, production_text: str):
        """Parsea una producción individual"""
        if not production_text or production_text == ';':
            return
        
        production_text = production_text.rstrip(';')
        
        # Dividir por ':'
        parts = production_text.split(':', 1)
        if len(parts) != 2:
            return
        
        left_side = parts[0].strip()
        right_side = parts[1].strip()
        
        # Dividir alternativas por '|'
        alternatives = [alt.strip() for alt in right_side.split('|')]
        
        for alternative in alternatives:
            if alternative:
                symbols = alternative.split()
                for symbol in symbols:
                    if ':' in symbol:
                        print(f"❗ Símbolo inválido detectado: '{symbol}' en producción: {left_side} -> {alternative}")
                production = Production(left_side, symbols)
                self.productions.append(production)

class SLR1Parser:
    """Implementa el algoritmo SLR(1)"""
    
    def __init__(self, tokens: List[str], productions: List[Production]):
        self.tokens = set(tokens)
        self.productions = productions
        self.non_terminals = set()
        self.terminals = set()
        self.start_symbol = None
        
        # Calcular símbolos
        self._calculate_symbols()
        
        # Agregar producción inicial
        if self.start_symbol:
            # Evitar conflicto con símbolos existentes
            augmented_start = "S'"
            while augmented_start in self.non_terminals:
                augmented_start += "'"
            start_production = Production(augmented_start, [self.start_symbol])
            self.productions.insert(0, start_production)
            self.non_terminals.add(augmented_start)  # <== ESTA LÍNEA ES CLAVE
            self.augmented_start = augmented_start
        else:
            raise ValueError("No se encontró símbolo inicial en la gramática")
        
        # Calcular conjuntos FIRST y FOLLOW
        self.first_sets = {}
        self.follow_sets = {}
        self._calculate_first_sets()
        self._calculate_follow_sets()
        
        # Construir autómata LR(0)
        self.states = []
        self.goto_table = {}
        self.action_table = {}
        self._build_lr0_automaton()
        self._build_slr1_table()
    
    def _calculate_symbols(self):
        """Calcula terminales y no-terminales"""
        # Paso 1: registrar no terminales (todos los lados izquierdos)
        for production in self.productions:
            self.non_terminals.add(production.left)

        # Asignar el símbolo inicial
        if not self.start_symbol:
            for production in self.productions:
                if not production.left.endswith("'"):
                    self.start_symbol = production.left
                    break

        # Paso 2: clasificar símbolos del lado derecho
        for production in self.productions:
            for symbol in production.right:
                if symbol in self.tokens:
                    self.terminals.add(symbol)
                elif symbol not in self.non_terminals:
                    # Podría ser epsilon o un error real, dependiendo del diseño
                    if symbol != 'ε':
                        print(f"Advertencia: Símbolo '{symbol}' no reconocido ni como token ni como no-terminal")

        
        # Agregar $ como terminal especial
        self.terminals.add('$')
    
    def _calculate_first_sets(self):
        """Calcula los conjuntos FIRST"""
        # Inicializar conjuntos FIRST
        for symbol in self.terminals:
            self.first_sets[symbol] = {symbol}
        
        for symbol in self.non_terminals:
            self.first_sets[symbol] = set()
        
        # Algoritmo para calcular FIRST
        changed = True
        while changed:
            changed = False
            for production in self.productions:
                first_before = len(self.first_sets[production.left])
                
                if not production.right:  # Producción epsilon
                    self.first_sets[production.left].add('ε')
                else:
                    # Para cada símbolo en el lado derecho
                    all_have_epsilon = True
                    for symbol in production.right:
                        # Agregar FIRST(symbol) - {ε} a FIRST(left)
                        first_symbol = self.first_sets.get(symbol, {symbol})
                        self.first_sets[production.left].update(first_symbol - {'ε'})
                        
                        # Si el símbolo no deriva ε, parar
                        if 'ε' not in first_symbol:
                            all_have_epsilon = False
                            break
                    
                    # Si todos los símbolos derivan ε, agregar ε
                    if all_have_epsilon:
                        self.first_sets[production.left].add('ε')
                
                if len(self.first_sets[production.left]) > first_before:
                    changed = True
    
    def _calculate_follow_sets(self):
        """Calcula los conjuntos FOLLOW"""
        # Inicializar conjuntos FOLLOW
        for symbol in self.non_terminals:
            self.follow_sets[symbol] = set()
        
        # FOLLOW del símbolo inicial contiene $
        if self.start_symbol:
            self.follow_sets[self.start_symbol].add('$')
        
        # Algoritmo para calcular FOLLOW
        changed = True
        while changed:
            changed = False
            for production in self.productions:
                for i, symbol in enumerate(production.right):
                    if symbol in self.non_terminals:
                        follow_before = len(self.follow_sets[symbol])
                        
                        # Símbolos después de la posición actual
                        beta = production.right[i+1:]
                        
                        if beta:
                            # Calcular FIRST(beta)
                            first_beta = self._first_of_sequence(beta)
                            self.follow_sets[symbol].update(first_beta - {'ε'})
                            
                            # Si ε ∈ FIRST(beta), agregar FOLLOW(left)
                            if 'ε' in first_beta:
                                self.follow_sets[symbol].update(self.follow_sets[production.left])
                        else:
                            # No hay símbolos después, agregar FOLLOW(left)
                            self.follow_sets[symbol].update(self.follow_sets[production.left])
                        
                        if len(self.follow_sets[symbol]) > follow_before:
                            changed = True
    
    def _first_of_sequence(self, sequence: List[str]) -> Set[str]:
        """Calcula FIRST de una secuencia de símbolos"""
        if not sequence:
            return {'ε'}
        
        result = set()
        all_have_epsilon = True
        
        for symbol in sequence:
            first_symbol = self.first_sets.get(symbol, {symbol})
            result.update(first_symbol - {'ε'})
            
            if 'ε' not in first_symbol:
                all_have_epsilon = False
                break
        
        if all_have_epsilon:
            result.add('ε')
        
        return result
    
    def _closure(self, items: Set[Item]) -> Set[Item]:
        """Calcula la clausura de un conjunto de items"""
        closure = set(items)
        added = True
        
        while added:
            added = False
            new_items = set()
            
            for item in closure:
                if not item.is_complete():
                    next_sym = item.next_symbol()
                    if next_sym in self.non_terminals:
                        # Agregar todos los items [A -> •α] donde A -> α es una producción
                        for production in self.productions:
                            if production.left == next_sym:
                                new_item = Item(production, 0)
                                if new_item not in closure:
                                    new_items.add(new_item)
                                    added = True
            
            closure.update(new_items)
        
        return closure
    
    def _goto(self, items: Set[Item], symbol: str) -> Set[Item]:
        """Calcula GOTO(I, X)"""
        goto_items = set()
        
        for item in items:
            if not item.is_complete() and item.next_symbol() == symbol:
                goto_items.add(item.advance())
        
        return self._closure(goto_items)
    
    def _build_lr0_automaton(self):
        """Construye el autómata LR(0)"""
        # Estado inicial
        if not self.productions:
            return
        
        initial_item = Item(self.productions[0], 0)  # S' -> •S
        initial_state = self._closure({initial_item})
        self.states = [initial_state]
        
        # Cola de estados por procesar
        queue = [0]
        state_map = {frozenset(initial_state): 0}
        
        while queue:
            state_idx = queue.pop(0)
            current_state = self.states[state_idx]
            
            # Encontrar todos los símbolos después del punto
            symbols = set()
            for item in current_state:
                if not item.is_complete():
                    symbols.add(item.next_symbol())
            
            # Para cada símbolo, calcular GOTO
            for symbol in symbols:
                goto_state = self._goto(current_state, symbol)
                
                if goto_state:
                    # Verificar si ya existe este estado
                    goto_frozen = frozenset(goto_state)
                    if goto_frozen in state_map:
                        # Estado ya existe
                        target_idx = state_map[goto_frozen]
                    else:
                        # Nuevo estado
                        target_idx = len(self.states)
                        self.states.append(goto_state)
                        state_map[goto_frozen] = target_idx
                        queue.append(target_idx)
                    
                    # Agregar transición
                    self.goto_table[(state_idx, symbol)] = target_idx
    
    def _build_slr1_table(self):
        """Construye la tabla SLR(1)"""
        self.action_table = {}

        for state_idx, state in enumerate(self.states):
            for item in state:
                if item.is_complete():
                    prod_idx = self.productions.index(item.production)

                    if item.production.left == self.augmented_start and len(item.production.right) == 1:
                        # Aceptación: S' -> expression .
                        self.action_table[(state_idx, '$')] = ('accept', None)
                    else:
                        # Reducción: A -> α .
                        for terminal in self.follow_sets.get(item.production.left, set()):
                            if (state_idx, terminal) in self.action_table:
                                print(f"Conflicto en estado {state_idx}, terminal {terminal}")
                            self.action_table[(state_idx, terminal)] = ('reduce', prod_idx)
                else:
                    # Desplazamiento: A -> α • a β
                    next_sym = item.next_symbol()
                    if next_sym in self.terminals and (state_idx, next_sym) in self.goto_table:
                        target_state = self.goto_table[(state_idx, next_sym)]
                        self.action_table[(state_idx, next_sym)] = ('shift', target_state)

    
    def print_automaton(self):
        """Imprime el autómata LR(0)"""
        print("=== AUTÓMATA LR(0) ===")
        for i, state in enumerate(self.states):
            print(f"\nEstado {i}:")
            for item in sorted(state, key=str):
                print(f"  {item}")
        
        print("\n=== TABLA GOTO ===")
        for (state, symbol), target in sorted(self.goto_table.items()):
            print(f"GOTO({state}, {symbol}) = {target}")
        
        print("\n=== TABLA DE ACCIÓN ===")
        for (state, symbol), (action, param) in sorted(self.action_table.items()):
            if action == 'shift':
                print(f"ACTION({state}, {symbol}) = shift {param}")
            elif action == 'reduce':
                print(f"ACTION({state}, {symbol}) = reduce {param}")
            else:
                print(f"ACTION({state}, {symbol}) = {action}")
    
    def print_first_follow(self):
        """Imprime los conjuntos FIRST y FOLLOW"""
        print("\n=== CONJUNTOS FIRST ===")
        for symbol in sorted(self.first_sets.keys()):
            print(f"FIRST({symbol}) = {{{', '.join(sorted(self.first_sets[symbol]))}}}")
        
        print("\n=== CONJUNTOS FOLLOW ===")
        for symbol in sorted(self.follow_sets.keys()):
            print(f"FOLLOW({symbol}) = {{{', '.join(sorted(self.follow_sets[symbol]))}}}")
    
    def parse_string(self, tokens: List[str]) -> bool:
        """Parsea una cadena de tokens"""
        print(f"\n=== PARSEANDO: {' '.join(tokens)} ===")
        
        stack = [0]  # Pila con estados
        tokens = tokens + ['$']  # Agregar fin de cadena
        position = 0
        
        print(f"{'Paso':<4} {'Pila':<15} {'Entrada':<15} {'Acción'}")
        print("-" * 50)
        
        step = 0
        while True:
            current_state = stack[-1]
            current_token = tokens[position] if position < len(tokens) else '$'
            
            print(f"{step:<4} {str(stack):<15} {' '.join(tokens[position:]):<15} ", end="")
            
            if (current_state, current_token) not in self.action_table:
                print("ERROR: Acción no definida")
                return False
            
            action, param = self.action_table[(current_state, current_token)]
            
            if action == 'shift':
                print(f"shift {param}")
                stack.append(param)
                position += 1
            elif action == 'reduce':
                production = self.productions[param]
                print(f"reduce {param} ({production})")
                
                # Remover símbolos de la pila
                for _ in range(len(production.right)):
                    if len(stack) > 1:
                        stack.pop()
                
                # Buscar nuevo estado con GOTO
                top_state = stack[-1]
                if (top_state, production.left) in self.goto_table:
                    new_state = self.goto_table[(top_state, production.left)]
                    stack.append(new_state)
                else:
                    print("ERROR: GOTO no definido")
                    return False
            elif action == 'accept':
                print("ACEPTAR")
                return True
            else:
                print(f"ERROR: Acción desconocida {action}")
                return False
            
            step += 1
            if step > 100:  # Evitar bucles infinitos
                print("ERROR: Demasiados pasos")
                return False

def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python yapar.py archivo.yalp [opciones]")
        print("Opciones:")
        print("  -l archivo.yalex    Archivo YALex para tokens")
        print("  -t 'cadena'         Parsear cadena de tokens")
        print("  -f archivo.txt      Parsear cadenas desde archivo")
        print("  -o salida          Archivo de salida (opcional)")
        sys.exit(1)
    
    yapar_file = sys.argv[1]
    yalex_file = None
    output_file = None
    
    try:
        # Parsear archivo YAPar
        parser = YALexProcessor()
        tokens, ignore_tokens, productions = parser.parse_file(yapar_file)
        
        # Procesar argumentos
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '-l' and i + 1 < len(sys.argv):
                yalex_file = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '-o' and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            else:
                break
        
        # Procesar YALex si se proporciona
        if yalex_file:
            yalex_processor = YALexProcessor()
            yalex_tokens = yalex_processor.process_yalex_file(yalex_file)
            if yalex_tokens:
                print(f"=== TOKENS DESDE YALEX ===")
                print(f"Tokens encontrados: {yalex_tokens}")
                # Combinar tokens de ambas fuentes
                all_tokens = list(set(tokens + yalex_tokens))
            else:
                all_tokens = tokens
        else:
            all_tokens = tokens
        
        print("=== TOKENS ===")
        print(f"Tokens: {all_tokens}")
        print(f"Ignorar: {ignore_tokens}")
        
        print("\n=== PRODUCCIONES ===")
        for i, prod in enumerate(productions):
            print(f"{i}: {prod}")
        
        # Crear parser SLR(1)
        try:
            slr_parser = SLR1Parser(all_tokens, productions)
            
            # Mostrar información
            slr_parser.print_first_follow()
            slr_parser.print_automaton()
            
            # Continuar procesando argumentos para parsing
            while i < len(sys.argv):
                if sys.argv[i] == '-t' and i + 1 < len(sys.argv):
                    # Parsear cadena de tokens
                    token_string = sys.argv[i + 1]
                    test_tokens = token_string.split()
                    result = slr_parser.parse_string(test_tokens)
                    print(f"Resultado: {'ACEPTADA' if result else 'RECHAZADA'}")
                    i += 2
                elif sys.argv[i] == '-f' and i + 1 < len(sys.argv):
                    # Parsear archivo de cadenas
                    strings_file = sys.argv[i + 1]
                    try:
                        with open(strings_file, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line and not line.startswith('#'):  # Ignorar comentarios
                                    test_tokens = line.split()
                                    print(f"\nLínea {line_num}: {line}")
                                    result = slr_parser.parse_string(test_tokens)
                                    print(f"Resultado: {'ACEPTADA' if result else 'RECHAZADA'}")
                    except FileNotFoundError:
                        print(f"Error: No se pudo encontrar el archivo {strings_file}")
                    i += 2
                else:
                    i += 1
                    
        except Exception as e:
            print(f"Error construyendo parser SLR(1): {e}")
            print("Verifique que la gramática sea válida para SLR(1)")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    def print_automaton(self):
        """Imprime el autómata LR(0)"""
        print("=== AUTÓMATA LR(0) ===")
        for i, state in enumerate(self.states):
            print(f"\nEstado {i}:")
            for item in sorted(state, key=str):
                print(f"  {item}")
        
        print("\n=== TABLA GOTO ===")
        for (state, symbol), target in sorted(self.goto_table.items()):
            print(f"GOTO({state}, {symbol}) = {target}")
        
        print("\n=== TABLA DE ACCIÓN ===")
        for (state, symbol), (action, param) in sorted(self.action_table.items()):
            if action == 'shift':
                print(f"ACTION({state}, {symbol}) = shift {param}")
            elif action == 'reduce':
                print(f"ACTION({state}, {symbol}) = reduce {param}")
            else:
                print(f"ACTION({state}, {symbol}) = {action}")
    
    def print_first_follow(self):
        """Imprime los conjuntos FIRST y FOLLOW"""
        print("\n=== CONJUNTOS FIRST ===")
        for symbol in sorted(self.first_sets.keys()):
            print(f"FIRST({symbol}) = {{{', '.join(sorted(self.first_sets[symbol]))}}}")
        
        print("\n=== CONJUNTOS FOLLOW ===")
        for symbol in sorted(self.follow_sets.keys()):
            print(f"FOLLOW({symbol}) = {{{', '.join(sorted(self.follow_sets[symbol]))}}}")
    
    def parse_string(self, tokens: List[str]) -> bool:
        """Parsea una cadena de tokens"""
        print(f"\n=== PARSEANDO: {' '.join(tokens)} ===")
        
        stack = [0]  # Pila con estados
        tokens = tokens + ['$']  # Agregar fin de cadena
        position = 0
        
        print(f"{'Paso':<4} {'Pila':<15} {'Entrada':<15} {'Acción'}")
        print("-" * 50)
        
        step = 0
        while True:
            current_state = stack[-1]
            current_token = tokens[position] if position < len(tokens) else '$'
            
            print(f"{step:<4} {str(stack):<15} {' '.join(tokens[position:]):<15} ", end="")
            
            if (current_state, current_token) not in self.action_table:
                print("ERROR: Acción no definida")
                return False
            
            action, param = self.action_table[(current_state, current_token)]
            
            if action == 'shift':
                print(f"shift {param}")
                stack.append(param)
                position += 1
            elif action == 'reduce':
                production = self.productions[param]
                print(f"reduce {param} ({production})")
                
                # Remover símbolos de la pila
                for _ in range(len(production.right)):
                    if len(stack) > 1:
                        stack.pop()
                
                # Buscar nuevo estado con GOTO
                top_state = stack[-1]
                if (top_state, production.left) in self.goto_table:
                    new_state = self.goto_table[(top_state, production.left)]
                    stack.append(new_state)
                else:
                    print("ERROR: GOTO no definido")
                    return False
            elif action == 'accept':
                print("ACEPTAR")
                return True
            else:
                print(f"ERROR: Acción desconocida {action}")
                return False
            
            step += 1
            if step > 100:  # Evitar bucles infinitos
                print("ERROR: Demasiados pasos")
                return False

def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python yapar.py archivo.yalp [-t cadena_tokens] [-f archivo_cadenas]")
        sys.exit(1)
    
    yapar_file = sys.argv[1]
    
    try:
        # Parsear archivo YAPar
        parser = YALexProcessor()
        tokens, ignore_tokens, productions = parser.parse_file(yapar_file)
        
        print("=== TOKENS ===")
        print(f"Tokens: {tokens}")
        print(f"Ignorar: {ignore_tokens}")
        
        print("\n=== PRODUCCIONES ===")
        for i, prod in enumerate(productions):
            print(f"{i}: {prod}")
        
        # Crear parser SLR(1)
        slr_parser = SLR1Parser(tokens, productions)
        
        # Mostrar información
        slr_parser.print_first_follow()
        slr_parser.print_automaton()
        
        # Procesar argumentos adicionales
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '-t' and i + 1 < len(sys.argv):
                # Parsear cadena de tokens
                token_string = sys.argv[i + 1]
                test_tokens = token_string.split()
                result = slr_parser.parse_string(test_tokens)
                print(f"Resultado: {'ACEPTADA' if result else 'RECHAZADA'}")
                i += 2
            elif sys.argv[i] == '-f' and i + 1 < len(sys.argv):
                # Parsear archivo de cadenas
                strings_file = sys.argv[i + 1]
                try:
                    with open(strings_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                test_tokens = line.split()
                                print(f"\nLínea {line_num}: {line}")
                                result = slr_parser.parse_string(test_tokens)
                                print(f"Resultado: {'ACEPTADA' if result else 'RECHAZADA'}")
                except FileNotFoundError:
                    print(f"Error: No se pudo encontrar el archivo {strings_file}")
                i += 2
            else:
                i += 1
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()