import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

export interface BenchmarkMetrics {
  dataset_name: string;
  sample_size: number;
  our_accuracy: number;
  our_correct: number;
  paper_accuracy: number;
  wilson_ci_95: number[];
  paper_ci_95: string;
  micro_precision: number;
  micro_recall: number;
  f1_score: number;
  specificity: number;
  mean_calibrated_confidence: number;
  mean_latency_ms: number;
  paper_latency: string;
}

export interface IterationProgressionItem {
  round: number;
  name: string;
  resolved_count: number;
  resolved_pct: number;
  running_accuracy: number;
  avg_latency: string;
}

export interface AblationItem {
  configuration: string;
  accuracy: number;
  delta: string;
  paper_target: number;
}

export interface PresetCase {
  id: string;
  title: string;
  category: string;
  clinical_text: string;
  gold_diagnosis: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  // Tabs: 'playground', 'tester', 'comparison', 'iterations', 'ablations'
  activeTab = signal<'playground' | 'tester' | 'comparison' | 'iterations' | 'ablations'>('playground');

  // Backend state
  apiUrl = 'http://localhost:8000';
  isBackendConnected = signal<boolean>(false);
  indexedPoints = signal<number>(7835);
  
  // Benchmark comparison data
  benchmarkData = signal<any>(null);
  isLoadingData = signal<boolean>(false);

  // Diagnostic Playground state
  presets = signal<PresetCase[]>([]);
  selectedPreset = signal<PresetCase | null>(null);
  clinicalInputText = signal<string>('');
  isDiagnosing = signal<boolean>(false);
  diagnosisResult = signal<any>(null);
  diagnosticError = signal<string | null>(null);

  // Test Runner state
  isTesting = signal<boolean>(false);
  selectedDataset = signal<'medqa' | 'pubmedqa' | 'medbullets'>('medqa');
  testCasesCount = signal<number>(10);
  testRunResults = signal<any>(null);
  testError = signal<string | null>(null);
  showExecutionLogs = signal<boolean>(true);

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.checkHealth();
    this.fetchBenchmarkResults();
    this.fetchPresets();
  }

  checkHealth() {
    this.http.get<any>(`${this.apiUrl}/api/health`).subscribe({
      next: (res) => {
        this.isBackendConnected.set(true);
        this.indexedPoints.set(res.qdrant_indexed_points || 7835);
      },
      error: () => {
        this.isBackendConnected.set(false);
      }
    });
  }

  fetchBenchmarkResults() {
    this.isLoadingData.set(true);
    this.http.get<any>(`${this.apiUrl}/api/benchmark/results`).subscribe({
      next: (res) => {
        this.benchmarkData.set(res);
        this.isLoadingData.set(false);
      },
      error: () => {
        // Fallback default state matching verified results
        this.benchmarkData.set(this.getDefaultBenchmarkData());
        this.isLoadingData.set(false);
      }
    });
  }

  fetchPresets() {
    this.http.get<PresetCase[]>(`${this.apiUrl}/api/cases/presets`).subscribe({
      next: (cases) => {
        this.presets.set(cases);
        if (cases.length > 0) {
          this.selectPreset(cases[0]);
        }
      },
      error: () => {
        const fallbacks = this.getDefaultPresets();
        this.presets.set(fallbacks);
        this.selectPreset(fallbacks[0]);
      }
    });
  }

  selectPreset(preset: PresetCase) {
    this.selectedPreset.set(preset);
    this.clinicalInputText.set(preset.clinical_text);
    this.diagnosisResult.set(null);
    this.diagnosticError.set(null);
  }

  runDiagnosis() {
    const text = this.clinicalInputText().trim();
    if (!text) {
      this.diagnosticError.set('Please enter or select a clinical vignette narrative.');
      return;
    }

    this.isDiagnosing.set(true);
    this.diagnosticError.set(null);
    this.diagnosisResult.set(null);

    this.http.post<any>(`${this.apiUrl}/api/diagnose`, { clinical_text: text }).subscribe({
      next: (res) => {
        this.diagnosisResult.set(res);
        this.isDiagnosing.set(false);
      },
      error: (err) => {
        this.isDiagnosing.set(false);
        this.diagnosticError.set(err.error?.detail || 'Failed to execute diagnosis pipeline. Ensure backend is running.');
      }
    });
  }

  runBenchmarkTest(dataset: 'medqa' | 'pubmedqa' | 'medbullets', count: number = 10, randomSample: boolean = true) {
    this.isTesting.set(true);
    this.selectedDataset.set(dataset);
    this.testCasesCount.set(count);
    this.testError.set(null);
    this.testRunResults.set(null);

    this.http.post<any>(`${this.apiUrl}/api/benchmark/run`, {
      dataset: dataset,
      cases_count: count,
      random_sample: randomSample
    }).subscribe({
      next: (res) => {
        this.testRunResults.set(res);
        this.isTesting.set(false);
      },
      error: (err) => {
        this.isTesting.set(false);
        this.testError.set(err.error?.detail || `Error running benchmark evaluation on ${dataset}. Ensure backend is running.`);
      }
    });
  }

  getDefaultBenchmarkData() {
    return {
      master_comparison: null,
      iteration_progression: null,
      ablation_study: null
    };
  }

  getDefaultPresets(): PresetCase[] {
    return [
      {
        id: "CASE_STEMI",
        title: "Acute Inferior Myocardial Infarction",
        category: "Cardiology",
        clinical_text: "Case ID: MEDQA_001\nClinical Case Vignette:\nA 54-year-old male presents to the ED with acute retrosternal crushing chest pain radiating to his left arm and jaw. He is diaphoretic, short of breath, with ST elevations in leads II, III, and aVF. Elevated cardiac troponin I.\n\nCandidate Diagnostic Options:\nA: Myocardial Infarction\nB: Aortic Dissection\nC: Gastroesophageal Reflux\nD: Costochondritis",
        gold_diagnosis: "Myocardial Infarction"
      },
      {
        id: "CASE_PE",
        title: "Acute Pulmonary Embolism",
        category: "Pulmonology",
        clinical_text: "Case ID: MEDQA_002\nClinical Case Vignette:\nA 32-year-old female presents with acute pleuritic chest pain and dyspnea after a 14-hour flight. Tachycardia (118 bpm), hypoxia (89% O2 sat), and unilateral right calf swelling and tenderness.\n\nCandidate Diagnostic Options:\nA: Pulmonary Embolism\nB: Pneumothorax\nC: Pneumonia\nD: Anxiety Disorder",
        gold_diagnosis: "Pulmonary Embolism"
      },
      {
        id: "CASE_DKA",
        title: "Diabetic Ketoacidosis",
        category: "Endocrinology",
        clinical_text: "Case ID: MEDQA_003\nClinical Case Vignette:\nA 19-year-old male with type 1 diabetes presents with vomiting, diffuse abdominal pain, confusion, Kussmaul respirations, fruity breath odor. Blood glucose 480 mg/dL, arterial pH 7.18, positive ketones.\n\nCandidate Diagnostic Options:\nA: Diabetic Ketoacidosis\nB: Hyperosmolar Hyperglycemic State\nC: Acute Pancreatitis\nD: Gastroenteritis",
        gold_diagnosis: "Diabetic Ketoacidosis"
      }
    ];
  }
}
