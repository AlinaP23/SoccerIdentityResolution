package de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.matchingRules.transfermarket_jokecamp;

import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.ErrorAnalysisPlayers;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.blockers.PlayerBlockerByBirthYear;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.blockers.PlayerBlockerByFirstLettersOfName;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.comparators.PlayerBirthDateComparatorLevenshtein;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.comparators.PlayerHeightComparator;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.comparators.PlayerNameComparatorLevenshtein;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.model.Player;
import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.model.PlayerXMLReader;
import de.uni_mannheim.informatik.dws.winter.matching.MatchingEngine;
import de.uni_mannheim.informatik.dws.winter.matching.MatchingEvaluator;
import de.uni_mannheim.informatik.dws.winter.matching.algorithms.RuleLearner;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.NoBlocker;
import de.uni_mannheim.informatik.dws.winter.matching.blockers.StandardRecordBlocker;
import de.uni_mannheim.informatik.dws.winter.matching.rules.WekaMatchingRule;
import de.uni_mannheim.informatik.dws.winter.model.Correspondence;
import de.uni_mannheim.informatik.dws.winter.model.HashedDataSet;
import de.uni_mannheim.informatik.dws.winter.model.MatchingGoldStandard;
import de.uni_mannheim.informatik.dws.winter.model.Performance;
import de.uni_mannheim.informatik.dws.winter.model.defaultmodel.Attribute;
import de.uni_mannheim.informatik.dws.winter.model.defaultmodel.FeatureVectorDataSet;
import de.uni_mannheim.informatik.dws.winter.model.defaultmodel.RecordCSVFormatter;
import de.uni_mannheim.informatik.dws.winter.model.io.CSVCorrespondenceFormatter;
import de.uni_mannheim.informatik.dws.winter.processing.Processable;

import java.io.File;

/**
 * Data Set Transfermarket ↔ Jokecamp
 * Learning Combination Rules for Players
 */
public class IR_weka_players
{

    static boolean WRITE_FEATURE_SET_FOR_EXTERNAL_TOOL = true;

    public static void main( String[] args ) throws Exception
    {

        // loading data
    	HashedDataSet<Player, Attribute> dataTransferMarket = new HashedDataSet<>();
        new PlayerXMLReader().loadFromXML(new File("data/input/transfermarket.xml"), "/clubs/club/players/player", dataTransferMarket);
        HashedDataSet<Player, Attribute> dataJokecamp = new HashedDataSet<>();
        new PlayerXMLReader().loadFromXML(new File("data/input/jokecamp-others.xml"), "/clubs/club/players/player", dataJokecamp);
        
        System.out.println("Sample from transfermarket others: " + dataTransferMarket.getRandomRecord());
        System.out.println("Sample from jokecamp: " + dataJokecamp.getRandomRecord());

        String options[] = new String[] {""};
        String modelType = "SimpleLogistic"; // using a logistic regression
        WekaMatchingRule<Player, Attribute> matchingRule = new WekaMatchingRule<>(0.9, modelType, options);
        
        // add comparators
        matchingRule.addComparator(new PlayerNameComparatorLevenshtein());
        matchingRule.addComparator(new PlayerBirthDateComparatorLevenshtein());
        matchingRule.addComparator(new PlayerHeightComparator());

        // create a blocker (blocking strategy)
        //StandardRecordBlocker<Player, Attribute> blocker = new StandardRecordBlocker<Player, Attribute>(new PlayerBlockerByBirthYear());
        //NoBlocker<Player, Attribute> blocker = new NoBlocker<>();
        StandardRecordBlocker<Player, Attribute> blocker = new StandardRecordBlocker<Player, Attribute>(new PlayerBlockerByFirstLettersOfName(4));



        // load the gold standard (test set)
        MatchingGoldStandard goldStandardForTraining = new MatchingGoldStandard();
        goldStandardForTraining.loadFromCSVFile(new File("data/goldstandard/gs_transfermarket_jokecamp_players.csv"));

        // train the matching rule's model
        RuleLearner<Player, Attribute> learner = new RuleLearner<>();
        learner.learnMatchingRule(dataTransferMarket, dataJokecamp, null, matchingRule, goldStandardForTraining);

        // Initialize Matching Engine
        MatchingEngine<Player, Attribute> engine = new MatchingEngine<>();

        // Execute the matching
        Processable<Correspondence<Player, Attribute>> correspondences = engine.runIdentityResolution(
                dataTransferMarket, dataJokecamp, null, matchingRule,
                blocker);

        // write the correspondences to the output file
        new CSVCorrespondenceFormatter().writeCSV(new File("data/output/tranfermarket_jokecamp_correspondences_players.csv"), correspondences);

        // evaluate your result
        MatchingEvaluator<Player, Attribute> evaluator = new MatchingEvaluator<Player, Attribute>(true);
        Performance perfTest = evaluator.evaluateMatching(correspondences.get(),
                goldStandardForTraining);
        new ErrorAnalysisPlayers().printFalsePositives(correspondences, goldStandardForTraining);
        new ErrorAnalysisPlayers().printFalseNegatives(dataTransferMarket, dataJokecamp, correspondences, goldStandardForTraining);
        // print the evaluation result
        System.out.println("Transfermarket ↔ Jokecamp");
        System.out
                .println(String.format(
                        "Precision: %.4f\nRecall: %.4f\nF1: %.4f",
                        perfTest.getPrecision(), perfTest.getRecall(),
                        perfTest.getF1()));

        if(WRITE_FEATURE_SET_FOR_EXTERNAL_TOOL) {

            System.out.println("Writing Features for an External Tool...");

            // generate feature data set for RapidMiner
            RuleLearner<Player, Attribute> learner2 = new RuleLearner<>();

            FeatureVectorDataSet features = learner2.generateTrainingDataForLearning(
                    dataTransferMarket, dataJokecamp, goldStandardForTraining, matchingRule, null
            );

            new RecordCSVFormatter().writeCSV(new File("data/output/tranfermarket_jokecamp_features.csv"), features);

            System.out.println("Finished Writing...");
        }

    }


}
