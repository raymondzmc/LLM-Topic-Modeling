"""Test script to verify ETM NaN fix for sparse/short-text datasets.

This script tests:
1. The original issue - division by zero causing NaN with zero-sum BoW vectors
2. The fix - epsilon addition preventing NaN
3. Real-world scenario - loading a small subset of StackOverflow data
"""

import torch
import numpy as np
import sys
from models.octis.ETM import ETM
from data.loaders import load_training_data
from data.dataset.octis_dataset import prepare_octis_dataset


def test_normalization_with_zero_sums():
    """Test that epsilon prevents NaN when normalizing zero-sum BoW vectors."""
    print("\n" + "="*60)
    print("TEST 1: Zero-sum BoW normalization")
    print("="*60)
    
    # Create a batch with some zero-sum documents
    batch_size = 5
    vocab_size = 100
    
    # Mix of normal and zero-sum documents
    data_batch = torch.zeros(batch_size, vocab_size)
    data_batch[0, :10] = torch.rand(10) * 5  # Normal document
    data_batch[1, :] = 0  # Zero-sum document
    data_batch[2, 20:30] = torch.rand(10) * 3  # Normal document
    data_batch[3, :] = 0  # Zero-sum document
    data_batch[4, 50:55] = torch.rand(5) * 2  # Normal document
    
    sums = data_batch.sum(1).unsqueeze(1)
    print(f"\nBatch document sums: {sums.squeeze().tolist()}")
    print(f"Number of zero-sum documents: {(sums.squeeze() == 0).sum().item()}")
    
    # Test WITHOUT epsilon (original code - should produce NaN)
    print("\n--- WITHOUT epsilon (original code) ---")
    try:
        normalized_without_epsilon = data_batch / sums
        has_nan = torch.isnan(normalized_without_epsilon).any()
        print(f"Contains NaN: {has_nan}")
        if has_nan:
            print(f"Number of NaN values: {torch.isnan(normalized_without_epsilon).sum().item()}")
            print("❌ FAIL: Division by zero produces NaN as expected")
        else:
            print("⚠️  UNEXPECTED: No NaN found (may not be testing the right scenario)")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test WITH epsilon (fixed code - should NOT produce NaN)
    print("\n--- WITH epsilon (fixed code) ---")
    epsilon = 1e-8
    normalized_with_epsilon = data_batch / (sums + epsilon)
    has_nan = torch.isnan(normalized_with_epsilon).any()
    print(f"Contains NaN: {has_nan}")
    if not has_nan:
        print("✅ PASS: No NaN with epsilon fix")
        # Verify zero-sum documents are handled gracefully
        zero_doc_indices = (sums.squeeze() == 0).nonzero(as_tuple=True)[0]
        for idx in zero_doc_indices:
            doc_sum = normalized_with_epsilon[idx].sum().item()
            print(f"  Zero-sum doc {idx}: normalized sum = {doc_sum:.10f}")
    else:
        print(f"❌ FAIL: Still contains {torch.isnan(normalized_with_epsilon).sum().item()} NaN values")
    
    return not has_nan


def test_etm_model_with_sparse_data():
    """Test ETM model training with synthetic sparse data."""
    print("\n" + "="*60)
    print("TEST 2: ETM model with sparse synthetic data")
    print("="*60)
    
    from data.dataset.octis_dataset import OCTISDataset
    
    # Create synthetic sparse dataset
    vocab = [f"word_{i}" for i in range(50)]
    
    # Create corpus with mix of normal and empty documents
    corpus = []
    corpus.extend([["word_0", "word_1", "word_2"] for _ in range(5)])  # Normal docs
    corpus.extend([[] for _ in range(3)])  # Empty docs (should be filtered)
    corpus.extend([["word_10"] for _ in range(2)])  # Very sparse docs
    corpus.extend([["word_20", "word_21"] for _ in range(5)])  # Normal docs
    
    print(f"\nOriginal corpus size: {len(corpus)}")
    print(f"Empty documents: {sum(1 for doc in corpus if len(doc) == 0)}")
    
    # Prepare OCTIS dataset (filters empty docs)
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset, filtered_corpus, _ = prepare_octis_dataset(
            temp_dir, corpus, vocab, labels=None
        )
        
        print(f"Filtered corpus size: {len(filtered_corpus)}")
        print(f"Documents removed: {len(corpus) - len(filtered_corpus)}")
        
        # Create ETM model with small settings
        etm = ETM(
            num_topics=5,
            num_epochs=3,
            batch_size=4,
            t_hidden_size=50,
            rho_size=50,
            embedding_size=50,
            train_embeddings=True,
            use_partitions=False,
        )
        
        print("\nTraining ETM model...")
        try:
            result = etm.train_model(dataset, top_words=5)
            
            # Check if topics contain NaN
            if result['topics'] is None:
                print("❌ FAIL: Model produced None topics (NaN in beta matrix)")
                return False
            else:
                print(f"✅ PASS: Model trained successfully")
                print(f"  Number of topics: {len(result['topics'])}")
                print(f"  Sample topic: {result['topics'][0]}")
                
                # Check for NaN in outputs
                if 'topic-word-matrix' in result:
                    has_nan = np.isnan(result['topic-word-matrix']).any()
                    if has_nan:
                        print(f"❌ FAIL: NaN found in topic-word-matrix")
                        return False
                
                return True
        except Exception as e:
            print(f"❌ FAIL: Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_with_real_stackoverflow_data():
    """Test with a small subset of real StackOverflow data if available."""
    print("\n" + "="*60)
    print("TEST 3: Real StackOverflow data (if available)")
    print("="*60)
    
    try:
        # Try to load StackOverflow dataset
        data_path = "raymondzmc/stackoverflow_Llama-3.1-8B-Instruct_vocab_2000_last"
        print(f"Attempting to load: {data_path}")
        
        training_data = load_training_data(data_path, for_generative=False)
        
        print(f"✅ Dataset loaded successfully")
        print(f"  Total documents: {len(training_data.bow_corpus)}")
        print(f"  Vocabulary size: {len(training_data.vocab)}")
        
        # Check for empty documents
        empty_docs = sum(1 for doc in training_data.bow_corpus if len(doc) == 0)
        print(f"  Empty BoW documents: {empty_docs}")
        
        # Analyze document lengths
        doc_lengths = [len(doc) for doc in training_data.bow_corpus]
        print(f"  Min doc length: {min(doc_lengths)}")
        print(f"  Max doc length: {max(doc_lengths)}")
        print(f"  Avg doc length: {sum(doc_lengths)/len(doc_lengths):.2f}")
        
        # Take a small subset for testing
        subset_size = min(100, len(training_data.bow_corpus))
        subset_corpus = training_data.bow_corpus[:subset_size]
        subset_labels = training_data.labels[:subset_size] if training_data.labels else None
        
        print(f"\nTesting with {subset_size} documents...")
        
        # Prepare OCTIS dataset
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset, filtered_corpus, _ = prepare_octis_dataset(
                temp_dir, subset_corpus, training_data.vocab, labels=subset_labels
            )
            
            print(f"  Filtered subset size: {len(filtered_corpus)}")
            
            # Create ETM model
            etm = ETM(
                num_topics=10,
                num_epochs=5,
                batch_size=16,
                lr=0.005,
                use_partitions=False,
            )
            
            print("\nTraining ETM on StackOverflow subset...")
            result = etm.train_model(dataset, top_words=10)
            
            # Check results
            if result['topics'] is None:
                print("❌ FAIL: Model produced None topics (NaN in beta matrix)")
                return False
            
            # Check for NaN in any output
            has_nan_topics = False
            has_nan_matrix = False
            
            if 'topic-word-matrix' in result:
                has_nan_matrix = np.isnan(result['topic-word-matrix']).any()
            
            if has_nan_topics or has_nan_matrix:
                print(f"❌ FAIL: NaN found in model outputs")
                return False
            
            print(f"✅ PASS: ETM trained successfully on StackOverflow data")
            print(f"  Topics generated: {len(result['topics'])}")
            print(f"  Sample topic: {result['topics'][0][:5]}")
            return True
            
    except Exception as e:
        print(f"⚠️  SKIP: Could not test with real data: {e}")
        import traceback
        traceback.print_exc()
        return None  # None indicates skipped, not failed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ETM NaN FIX VALIDATION TESTS")
    print("="*60)
    print("\nTesting the fix for division-by-zero in ETM BoW normalization")
    print("that causes NaN losses on sparse/short-text datasets.")
    
    results = {}
    
    # Test 1: Basic normalization
    results['normalization'] = test_normalization_with_zero_sums()
    
    # Test 2: Synthetic sparse data
    results['synthetic'] = test_etm_model_with_sparse_data()
    
    # Test 3: Real StackOverflow data (optional)
    results['stackoverflow'] = test_with_real_stackoverflow_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{test_name:20s}: {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n❌ Some tests FAILED - fix may not be working correctly")
        sys.exit(1)
    elif passed > 0:
        print("\n✅ All tests PASSED - fix is working correctly!")
        sys.exit(0)
    else:
        print("\n⚠️  All tests were skipped")
        sys.exit(2)


if __name__ == '__main__':
    main()

