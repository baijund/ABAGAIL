package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;


/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
    private static final int N = 25;
    
    private static String RHC_CSV = "";
    private static String RHC_CSV_PATH = "";
    private static String SA_CSV_PATH = "";
    private static String SA_CSV = "";
    private static String GA_CSV_PATH = "";
    private static String GA_CSV = "";
    private static String MIMIC_CSV_PATH = "";
    private static String MIMIC_CSV = "";
    
    private static int RHC_COUNT = 20;
    private static int SA_COUNT = 20;
    private static int GA_COUNT = 20;
    private static int MIMIC_COUNT = 20;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        FixedIterationTrainer fit;
        
        for(int i=0;i<RHC_COUNT;i++){
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
	        fit = new FixedIterationTrainer(rhc, 200000);
	        fit.train();
	        System.out.println("RHC"+i+" : "+ef.value(rhc.getOptimal()));
	        
	        
        }
        
        for(int i=0;i<SA_COUNT;i++){
	        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
	        fit = new FixedIterationTrainer(sa, 200000);
	        fit.train();
	        System.out.println("SA"+i+": "+ef.value(sa.getOptimal()));
	        
        }
        
        
        for(int i=0; i<GA_COUNT; i++){
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
	        fit = new FixedIterationTrainer(ga, 1000);
	        fit.train();
	        System.out.println("GA"+i+": "+ef.value(ga.getOptimal()));
        }
        
        for(int i=0;i<MIMIC_COUNT;i++){
	        MIMIC mimic = new MIMIC(200, 5, pop);
	        fit = new FixedIterationTrainer(mimic, 1000);
	        fit.train();
	        System.out.println("MIMIC"+i+": "+ef.value(mimic.getOptimal()));
        }
    }
}
