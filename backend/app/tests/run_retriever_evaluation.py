#!/usr/bin/env python3
"""
Retriever Evaluation Runner for RAG Application

This script allows you to evaluate different retriever configurations
in your RAG application using the RAGAS framework. You can customize
the retriever settings, chunking strategies, and evaluation metrics.

Usage:
    python run_retriever_evaluation.py
    python run_retriever_evaluation.py --quick    # Run a quick evaluation with fewer configs
    python run_retriever_evaluation.py --custom   # Use your own app's retriever settings
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluate import (
    evaluate_retriever_configurations,
    create_results_summary,
    save_evaluation_results,
    base_docs,
    eval_dataset
)


def get_quick_configs():
    """Get a smaller set of configurations for quick evaluation."""
    retriever_configs = {
        "similarity_k2": {"search_type": "similarity", "k": 2},
        "similarity_k5": {"search_type": "similarity", "k": 5},
        "mmr_k3": {"search_type": "mmr", "k": 3},
    }
    
    splitter_configs = {
        "small": {"chunk_size": 250, "chunk_overlap": 50},
        "medium": {"chunk_size": 500, "chunk_overlap": 100},
    }
    
    return retriever_configs, splitter_configs


def get_comprehensive_configs():
    """Get a comprehensive set of configurations for thorough evaluation."""
    retriever_configs = {
        "similarity_k1": {"search_type": "similarity", "k": 1},
        "similarity_k2": {"search_type": "similarity", "k": 2},
        "similarity_k3": {"search_type": "similarity", "k": 3},
        "similarity_k5": {"search_type": "similarity", "k": 5},
        "similarity_k10": {"search_type": "similarity", "k": 10},
        
        "mmr_k2": {"search_type": "mmr", "k": 2},
        "mmr_k3": {"search_type": "mmr", "k": 3},
        "mmr_k5": {"search_type": "mmr", "k": 5},
        
        "threshold_05": {
            "search_type": "similarity_score_threshold",
            "score_threshold": 0.5,
            "k": 10
        },
        "threshold_07": {
            "search_type": "similarity_score_threshold",
            "score_threshold": 0.7,
            "k": 10
        },
    }
    
    splitter_configs = {
        "tiny": {"chunk_size": 150, "chunk_overlap": 25},
        "small": {"chunk_size": 250, "chunk_overlap": 50},
        "medium": {"chunk_size": 500, "chunk_overlap": 100},
        "large": {"chunk_size": 1000, "chunk_overlap": 200},
        "xlarge": {"chunk_size": 1500, "chunk_overlap": 300},
    }
    
    return retriever_configs, splitter_configs


def get_app_configs():
    """Get configurations that match your application's current settings."""
    # Based on your app's settings from .env and service.py
    retriever_configs = {
        "app_default": {"search_type": "similarity", "k": 5},  # from service.py
        "app_optimized_k3": {"search_type": "similarity", "k": 3},
        "app_optimized_k7": {"search_type": "similarity", "k": 7},
        "app_mmr": {"search_type": "mmr", "k": 5},
    }
    
    splitter_configs = {
        "app_default": {"chunk_size": 1000, "chunk_overlap": 200},  # from .env
        "app_smaller": {"chunk_size": 500, "chunk_overlap": 100},
        "app_larger": {"chunk_size": 1500, "chunk_overlap": 300},
    }
    
    return retriever_configs, splitter_configs


def analyze_results(results, summary_df):
    """Provide detailed analysis of the evaluation results."""
    print("\n🔍 DETAILED ANALYSIS")
    print("=" * 50)
    
    if summary_df.empty:
        print("❌ No results to analyze")
        return
    
    # Best configuration overall
    best_config = summary_df.iloc[0]
    print(f"\n🏆 Best Overall Configuration: {best_config['config_name']}")
    print(f"   Search Type: {best_config['search_type']}")
    print(f"   K: {best_config['k']}")
    print(f"   Chunk Size: {best_config['chunk_size']}")
    print(f"   Composite Score: {best_config['composite_score']:.4f}")
    
    # Best by search type
    print(f"\n📈 Best by Search Type:")
    for search_type in summary_df['search_type'].unique():
        best_for_type = summary_df[summary_df['search_type'] == search_type].iloc[0]
        print(f"   {search_type}: {best_for_type['config_name']} "
              f"(score: {best_for_type['composite_score']:.4f})")
    
    # Best by chunk size
    print(f"\n📏 Best by Chunk Size:")
    for chunk_size in sorted(summary_df['chunk_size'].unique()):
        best_for_chunk = summary_df[summary_df['chunk_size'] == chunk_size].iloc[0]
        print(f"   {chunk_size}: {best_for_chunk['config_name']} "
              f"(score: {best_for_chunk['composite_score']:.4f})")
    
    # Metric analysis
    print(f"\n📊 Metric Analysis:")
    metrics = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']
    for metric in metrics:
        if not summary_df[metric].isnull().all():
            best_metric = summary_df.loc[summary_df[metric].idxmax()]
            print(f"   Best {metric}: {best_metric['config_name']} ({best_metric[metric]:.4f})")
        else:
            print(f"   Best {metric}: N/A (all NaN values)")


def main():
    parser = argparse.ArgumentParser(description="Run retriever evaluation for RAG application")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "app"], 
                       default="quick",
                       help="Evaluation mode: quick, comprehensive, or app-specific")
    parser.add_argument("--output-dir", default="./", 
                       help="Directory to save results")
    parser.add_argument("--config-file", 
                       help="JSON file with custom retriever and splitter configurations")

    args = parser.parse_args()

    print(f"🚀 Starting Retriever Evaluation in {args.mode} mode")
    print("=" * 60)

    # Get configurations based on mode
    if args.config_file and os.path.exists(args.config_file):
        print(f"📁 Loading custom configurations from {args.config_file}")
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
        retriever_configs = custom_config.get("retriever_configs", {})
        splitter_configs = custom_config.get("splitter_configs", {})
    elif args.mode == "quick":
        retriever_configs, splitter_configs = get_quick_configs()
    elif args.mode == "comprehensive":
        retriever_configs, splitter_configs = get_comprehensive_configs()
    elif args.mode == "app":
        retriever_configs, splitter_configs = get_app_configs()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print(f"📋 Configuration Summary:")
    print(f"   Retriever configs: {len(retriever_configs)}")
    print(f"   Splitter configs: {len(splitter_configs)}")
    print(f"   Total combinations: {len(retriever_configs) * len(splitter_configs)}")
    
    # Run evaluation
    results = evaluate_retriever_configurations(
        base_docs=base_docs,
        eval_dataset=eval_dataset,
        retriever_configs=retriever_configs,
        splitter_configs=splitter_configs
    )
    
    # Create output filenames with mode prefix
    results_file = os.path.join(args.output_dir, f"retriever_evaluation_{args.mode}_results.json")
    summary_file = os.path.join(args.output_dir, f"retriever_evaluation_{args.mode}_summary.csv")
    
    # Save results
    save_evaluation_results(results, results_file)
    
    # Create and save summary
    summary_df = create_results_summary(results)
    if not summary_df.empty:
        summary_df.to_csv(summary_file, index=False)
        print(f"💾 Summary saved to {summary_file}")
        
        # Display top configurations
        print(f"\n📊 TOP 10 CONFIGURATIONS")
        print("=" * 60)
        top_configs = summary_df.head(10)
        print(top_configs[[
            "config_name", "search_type", "k", "chunk_size",
            "context_precision", "faithfulness", "answer_relevancy", "composite_score"
        ]].to_string(index=False, float_format="%.3f"))
        
        # Detailed analysis
        analyze_results(results, summary_df)
        
    else:
        print("❌ No successful evaluations to display")
    
    print(f"\n✅ Evaluation completed!")
    print(f"\nFiles generated:")
    print(f"- {results_file}")
    if not summary_df.empty:
        print(f"- {summary_file}")


if __name__ == "__main__":
    main()
