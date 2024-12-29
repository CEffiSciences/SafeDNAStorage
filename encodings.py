from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
from Bio.Seq import Seq
from tqdm import tqdm
from reedsolo import RSCodec
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
import concurrent
import multiprocessing
from PIL import Image
import io
import base64
import time
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def binary_string_to_bytes(
    binary_string: str
) -> bytes:
    """Convert binary string to bytes, padding to nearest byte"""
    if not all(bit in '01' for bit in binary_string):
        raise ValueError("Input string must contain only '0' and '1' characters")
    
    padded_binary = binary_string.zfill((len(binary_string) + 7) // 8 * 8)
    num_bytes = len(padded_binary) // 8
    return int(padded_binary, 2).to_bytes(num_bytes, byteorder='big')

def binary_to_ternary(
    binary_str: str
) -> str:
    """Convert binary string to base-3 representation"""
    if binary_str == '0':
        return '0'
    
    value = 0
    for bit in binary_str:
        value = value * 2 + int(bit)
        
    if value == 0:
        return '0'
        
    ternary = []
    while value:
        ternary.append(str(value % 3))
        value //= 3
        
    return ''.join(reversed(ternary))

def ternary_to_binary(
    ternary_str: str,
    target_length: int = None
) -> str:
    """Convert base-3 string to binary, optionally padding to target length"""
    if ternary_str == '0':
        return '0' if target_length is None else '0' * target_length
    
    value = 0
    for digit in ternary_str:
        value = value * 3 + int(digit)
        
    if value == 0:
        return '0' if target_length is None else '0' * target_length
        
    binary = []
    while value:
        binary.append(str(value % 2))
        value //= 2
    
    result = ''.join(reversed(binary))
    
    if target_length is not None:
        result = '0' * (target_length - len(result)) + result
        
    return result

class DNAStorage:
    """Handles DNA encoding/decoding with error correction and nucleotide mapping"""
    
    def __init__(self,
        ecc_symbols: int = 10
    ):
        self.rs = RSCodec(ecc_symbols)
        # Mapping to avoid repeated nucleotides
        self.encode_mapping = {
            'A': {0: 'C', 1: 'G', 2: 'T'},
            'C': {0: 'G', 1: 'T', 2: 'A'},
            'G': {0: 'T', 1: 'A', 2: 'C'},
            'T': {0: 'A', 1: 'C', 2: 'G'}
        }
        self.reverse_encoding = {
            'A': {'C': 0, 'G': 1, 'T': 2},
            'C': {'G': 0, 'T': 1, 'A': 2},
            'G': {'T': 0, 'A': 1, 'C': 2},
            'T': {'A': 0, 'C': 1, 'G': 2}
        }

    def encode_text_to_bits(self,
        text: str
    ) -> str:
        """Convert text to binary with Reed-Solomon error correction"""
        binary_data = text.encode('utf-8')
        encoded_binary = self.rs.encode(binary_data)
        return ''.join(format(byte, '08b') for byte in encoded_binary)
    
    def encode_bits_to_DNA(self,
        bits: str
    ) -> str:
        """Convert binary to DNA using nucleotide mapping that avoids repeats"""
        bits = list(bits)
        bits[0] = '1'
        bits = "".join(bits)

        ternary_seq = binary_to_ternary(bits)
        ternary_seq = "0" * (9 - (len(ternary_seq) % 9)) + ternary_seq

        dna = ""
        previous_nucleotide = "A"
        for ternary_digit in str(ternary_seq):
            dna += self.encode_mapping[previous_nucleotide][int(ternary_digit)]
            previous_nucleotide = dna[-1]

        return dna

    def decode(self,
        dna: str
    ) -> str:
        """Convert DNA back to original text through ternary and binary"""
        ternary = ""
        previous_nucleotide = "A"
        for nucleotide in dna:
            ternary += str(self.reverse_encoding[previous_nucleotide][nucleotide])
            previous_nucleotide = nucleotide

        bits = ternary_to_binary(ternary)
        dividend = (len(bits) % 8)
        bits = bits[dividend:] if dividend > 0 else bits
        
        decoded = self.rs.decode(binary_string_to_bytes(bits))[0]
        return decoded.decode('utf-8')

    def hamming_distance(self,
        dna1: str,
        dna2: str
    ) -> float:
        """Calculate normalized Hamming distance between two DNA sequences"""
        return sum(n1 != n2 for n1, n2 in zip(dna1, dna2)) / len(dna1)

    def distance_from_list(self,
        dna: str,
        dna_lst: List[str]
    ) -> float:
        """Calculate average Hamming distance from a DNA sequence to a list of sequences"""
        return sum([
            self.hamming_distance(dna, dna_item)
            for dna_item in dna_lst
        ]) / len(dna_lst)

    def generate_distant_encodings(self,
        text: str,
        n_variants: int,
        mutations_per_variant: int = 1,
        best_of: int = 100,
        max_attempts: int = 100
    ) -> List[str]:
        """Generate diverse DNA encodings maximizing Hamming distance between variants"""
        base_encoding = self.encode_text_to_bits(text)

        def mutate(bits: str, n_mutations: int) -> str:
            positions = random.sample(range(len(bits)), n_mutations)
            bits_list = list(bits)
            for pos in positions:
                bits_list[pos] = '0' if bits_list == '1' else '1'
            return ''.join(bits_list)
        
        variants = []
        pbar = tqdm(total=n_variants)
        attempts = 0
        while len(variants) < n_variants and attempts < max_attempts:
            candidates = []
            for _ in range(best_of):
                candidate = mutate(base_encoding, mutations_per_variant)
                try:
                    decoded = self.decode(self.encode_bits_to_DNA(candidate))
                    if decoded == text and candidate not in variants:
                        if len(variants) == 0:
                            variants.append(candidate)
                            continue
                        else:
                            candidates.append(candidate)
                except Exception as e:
                    print(f"Error: {e}")

            if len(candidates) < 1:
                attempts += 1
            else:
                variants.append(
                    max(candidates,
                        key=lambda x: self.distance_from_list(x, variants)
                    )
                )
                pbar.update()
                pbar.display()

        return [self.encode_bits_to_DNA(variant) for variant in variants]

def generate_variations(
    text: str, 
    n_variants: int = 100, 
    ecc_symbols: float = 25,
    best_of: int = 5
) -> Tuple[List[str], DNAStorage]:

    codec = DNAStorage(ecc_symbols=ecc_symbols)
    mutations = ecc_symbols // 2

    return codec.generate_distant_encodings(text, n_variants, mutations, best_of=best_of), codec


@dataclass
class ProteinResult:
    """Result of protein toxicity analysis"""
    sequence: str
    is_toxic: bool
    score: float
    source_dna: str

class ToxinPredictor:
    """Predicts protein toxicity using ML model and protein features"""
    
    def __init__(self,
        model_path: str,
        threshold: float = 0.38
    ):
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def aac_comp(self,
        sequences: List[str]
    ) -> pd.DataFrame:
        """Calculate amino acid composition features"""
        std = list("ACDEFGHIKLMNPQRSTVWY")
        df1 = pd.DataFrame(sequences, columns=["Seq"])
        dd = []
        for j in df1['Seq']:
            cc = []
            for i in std:
                count = 0
                for k in j:
                    temp1 = k
                    if temp1 == i:
                        count += 1
                    composition = (count/len(j))*100
                cc.append(composition)
            dd.append(cc)
        df2 = pd.DataFrame(dd)
        head = []
        for mm in std:
            head.append('AAC_'+mm)
        df2.columns = head
        return df2

    def dpc_comp(self,
        sequences: List[str],
        q: int = 1
    ) -> pd.DataFrame:
        """Calculate dipeptide composition features"""
        std = list("ACDEFGHIKLMNPQRSTVWY")
        df1 = pd.DataFrame(sequences, columns=["Seq"])
        zz = df1.Seq
        dd = []
        for i in range(0,len(zz)):
            cc = []
            for j in std:
                for k in std:
                    count = 0
                    temp = j+k
                    for m3 in range(0,len(zz[i])-q):
                        b = zz[i][m3:m3+q+1:q]
                        b.upper()
                        if b == temp:
                            count += 1
                        composition = (count/(len(zz[i])-(q)))*100
                    cc.append(composition)
            dd.append(cc)
        df3 = pd.DataFrame(dd)
        head = []
        for s in std:
            for u in std:
                head.append("DPC"+str(q)+"_"+s+u)
        df3.columns = head
        return df3

    def predict(self,
        sequences: List[str]
    ) -> List[Tuple[float, bool]]:
        """Predict toxicity scores and binary classifications"""
        features = np.concatenate([
            self.aac_comp(sequences).values,
            self.dpc_comp(sequences).values
        ], axis=1)
        scores = self.model.predict_proba(features)[:, 1]
        return [(float(score), score >= self.threshold) for score in scores]

class DNAAnalyzer:
    """Analyzes DNA sequences for potential toxic proteins"""
    
    def __init__(self,
        predictor: ToxinPredictor
    ):
        self.predictor = predictor

    def _get_protein_variants(self,
        dna: str
    ) -> List[Tuple[str, str]]:
        """Extract all possible protein sequences from DNA and its complement"""
        dna_seq = Seq(dna)
        
        variants = [
            (dna, "Original DNA"),
            (str(dna_seq.complement()), "Complementary DNA"),
            (str(dna_seq[::-1]), "Reverse DNA"),
            (str(dna_seq.reverse_complement()), "Reverse Complementary DNA")
        ]
        
        results = []
        for variant_dna, desc in variants:
            protein = str(Seq(variant_dna).translate(to_stop=True))
            if len(protein) >= 1:
                results.append((protein, desc))
                
        return results

    def analyze_dna(self,
        dna: str
    ) -> List[ProteinResult]:
        """Analyze a single DNA sequence for potential toxic proteins"""
        variants = self._get_protein_variants(dna)
        
        if not variants:
            return []
            
        sequences = [v[0] for v in variants]
        predictions = self.predictor.predict(sequences)
        
        return [
            ProteinResult(
                sequence=seq,
                is_toxic=is_toxic,
                score=score,
                source_dna=source
            )
            for (seq, source), (score, is_toxic) 
            in zip(variants, predictions)
        ]
    
    def analyze_dna_lst(self,
        dnas: List[str]
    ) -> List[List[ProteinResult]]:
        """Analyze multiple DNA sequences in batch"""
        dna_variants = [self._get_protein_variants(dna) for dna in dnas]
        sequences = [[v[0] for v in variants] for variants in dna_variants]
        sequences = np.concatenate(sequences).tolist()
        
        predictions = self.predictor.predict(sequences=sequences)
        predictions = [predictions[i:i+4] for i in range(0, len(dna_variants) * 4, 4)]

        return [
            [
                ProteinResult(
                    sequence=seq,
                    is_toxic=is_toxic,
                    score=score,
                    source_dna=source
                )
                for (seq, source), (score, is_toxic) in zip(variants, pred)
            ] for variants, pred in zip(dna_variants, predictions)
        ]

class ChunkedDNAStorage:
    """Handles large-scale DNA storage operations by chunking data"""
    
    def __init__(self,
        ecc_symbols: int = 25
    ):
        self.ecc_symbols = ecc_symbols
        self.pad_char = "\x1F"
        self.codec = DNAStorage(ecc_symbols=self.ecc_symbols)

    def decode_text_chunks(self,
        dna_sequences: List[str]
    ) -> str:
        """Decode DNA sequences back to original text"""
        return ''.join(
            self.codec.decode(dna).rstrip(self.pad_char)
            for dna in dna_sequences
        )
    
    def encode_image_to_text(self,
        image_path: str
    ) -> str:
        """Convert image to base64 text representation"""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def decode_image_from_text(self,
        text: str
    ) -> bytes:
        """Convert base64 text back to image bytes"""
        return base64.b64decode(text)

    def save_image(self,
        image_data: bytes,
        output_path: str
    ):
        """Save image bytes to file"""
        with open(output_path, 'wb') as f:
            f.write(image_data)

def process_chunk(
    chunk: str,
    analyzer: DNAAnalyzer,
    initial_variants: int,
    max_variants: int,
    variant_increment: int,
    best_of: int,
    ecc_symbols: int,
    chunk_num: int,
    total_chunks: int
) -> Tuple[str, List[ProteinResult]]:
    """Process a single chunk to find safe encoding"""
    print(f"\nProcessing chunk {chunk_num}/{total_chunks}: '{chunk}'")
    n_variants = initial_variants
    
    while n_variants <= max_variants:
        print(f"Trying with {n_variants} variants...")
        
        try:
            variations, _ = generate_variations(
                chunk, 
                n_variants=n_variants,
                ecc_symbols=ecc_symbols,
                best_of=best_of
            )
            
            chunk_results = []
            analyzer_results = analyzer.analyze_dna_lst(variations)
            for i, protein_results in tqdm(enumerate(analyzer_results),
                                        desc=f"Analyzing chunk {chunk_num}"):
                if protein_results and not any(r.is_toxic for r in protein_results):
                    avg_score = sum(r.score for r in protein_results) / len(protein_results)
                    chunk_results.append((variations[i], protein_results, avg_score))
            
            if chunk_results:
                best_dna, best_results, _ = min(chunk_results, key=lambda x: x[2])
                print(f"Found safe encoding for chunk {chunk_num}")
                return best_dna, best_results
            
            n_variants += variant_increment
            print(f"No safe encoding found, increasing to {n_variants} variants")
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            n_variants += variant_increment
    
    raise ValueError(f"No safe encoding found for chunk: '{chunk}'")

def find_safe_encoding_chunked(
    text: str, 
    analyzer: DNAAnalyzer,
    chunk_size: int = 15,
    initial_variants: int = 20,
    max_variants: int = 100,
    variant_increment: int = 20,
    best_of: int = 5,
    ecc_symbols: int = 25,
    max_workers: int = None
) -> Tuple[List[str], List[List[ProteinResult]]]:
    """Find safe DNA encodings for chunked text with parallel processing"""
    max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
    dna_storage = ChunkedDNAStorage(ecc_symbols=ecc_symbols)
    
    chunks = [
        text[i:i + chunk_size].ljust(chunk_size, dna_storage.pad_char)
        for i in range(0, len(text), chunk_size)
    ]
    
    print(f"Split text into {len(chunks)} chunks")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                process_chunk,
                chunk,
                analyzer,
                initial_variants,
                max_variants,
                variant_increment,
                best_of,
                ecc_symbols,
                i + 1,
                len(chunks)
            ): i for i, chunk in enumerate(chunks)
        }
        
        safe_dnas = [None] * len(chunks)
        all_results = [None] * len(chunks)
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                dna, results = future.result()
                safe_dnas[chunk_idx] = dna
                all_results[chunk_idx] = results
            except Exception as e:
                raise ValueError(f"Failed to process chunk {chunk_idx + 1}: {e}")
    
    return safe_dnas, all_results

def process_image(
    image_path: str,
    output_path: str,
    dna_output_path: str,
    analyzer: DNAAnalyzer,
    chunk_size: int = 100,
    initial_variants: int = 20,
    max_variants: int = 100,
    variant_increment: int = 20,
    best_of: int = 5,
    ecc_symbols: int = 25,
    max_workers: int = None
) -> Tuple[List[str], List[List[ProteinResult]]]:
    """Process image through DNA encoding with safety checks"""
    start_time = time.time()
    process_times = {}
    
    dna_storage = ChunkedDNAStorage(ecc_symbols=ecc_symbols)
    
    print(f"Processing image: {image_path}")
    
    img_start = time.time()
    image_text = dna_storage.encode_image_to_text(image_path)
    process_times['image_to_text'] = time.time() - img_start
    
    encode_start = time.time()
    safe_dnas, results = find_safe_encoding_chunked(
        text=image_text,
        analyzer=analyzer,
        chunk_size=chunk_size,
        initial_variants=initial_variants,
        max_variants=max_variants,
        variant_increment=variant_increment,
        best_of=best_of,
        ecc_symbols=ecc_symbols,
        max_workers=max_workers
    )
    process_times['dna_encoding'] = time.time() - encode_start
    
    decode_start = time.time()
    decoded_text = dna_storage.decode_text_chunks(safe_dnas)
    image_data = dna_storage.decode_image_from_text(decoded_text)
    dna_storage.save_image(image_data, output_path)
    process_times['decoding_and_save'] = time.time() - decode_start
    
    total_time = time.time() - start_time
    process_times['total_time'] = total_time
    
    # Export DNA data
    dna_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'output_path': output_path,
            'parameters': {
                'chunk_size': chunk_size,
                'initial_variants': initial_variants,
                'max_variants': max_variants,
                'variant_increment': variant_increment,
                'best_of': best_of,
                'ecc_symbols': ecc_symbols,
                'max_workers': max_workers
            },
            'timing': process_times,
            'statistics': {
                'num_sequences': len(safe_dnas),
                'total_dna_length': sum(len(dna) for dna in safe_dnas),
                'original_text_length': len(image_text)
            }
        },
        'dna_sequences': safe_dnas
    }
    
    with open(dna_output_path, 'w') as f:
        json.dump(dna_data, f, indent=2)
    
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Image to text: {process_times['image_to_text']:.2f} seconds")
    print(f"DNA encoding: {process_times['dna_encoding']:.2f} seconds")
    print(f"Decoding and saving: {process_times['decoding_and_save']:.2f} seconds")
    print(f"DNA sequences saved to: {dna_output_path}")
    
    return safe_dnas, results


# Example usage:
if __name__ == "__main__":
    predictor = ToxinPredictor(
        model_path="model/toxinpred3.0_model.pkl",
        threshold=0.38
    )
    analyzer = DNAAnalyzer(predictor)

    # Cat image example
    # safe_dnas, results = process_image(
    #     image_path="input.jpg",
    #     output_path="output.jpg",
    #     dna_output_path="dna_sequences.json",
    #     analyzer=analyzer,
    #     chunk_size=100,
    #     initial_variants=20,
    #     max_variants=200,
    #     variant_increment=20,
    #     best_of=5,
    #     ecc_symbols=40
    # )

    data_to_encode = "This is a demo of how the encoding is done"
    safe_dnas = find_safe_encoding_chunked(
        text=data_to_encode,
        analyzer=analyzer,
        chunk_size=100,
    )

    print("Original:", data_to_encode)
    print("DNA Encoding:", safe_dnas)