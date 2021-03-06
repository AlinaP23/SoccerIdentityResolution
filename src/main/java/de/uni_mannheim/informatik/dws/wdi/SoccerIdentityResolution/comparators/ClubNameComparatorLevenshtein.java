package de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.comparators;

import de.uni_mannheim.informatik.dws.wdi.SoccerIdentityResolution.model.Club;
import de.uni_mannheim.informatik.dws.winter.matching.rules.Comparator;
import de.uni_mannheim.informatik.dws.winter.model.Correspondence;
import de.uni_mannheim.informatik.dws.winter.model.Matchable;
import de.uni_mannheim.informatik.dws.winter.model.defaultmodel.Attribute;
import de.uni_mannheim.informatik.dws.winter.similarity.string.LevenshteinSimilarity;


public class ClubNameComparatorLevenshtein implements Comparator<Club, Attribute> {

	private static final long serialVersionUID = 1L;
	private LevenshteinSimilarity sim = new LevenshteinSimilarity();
	private boolean lowerCase = false;

	public ClubNameComparatorLevenshtein() {
		super();
	}
	
	public ClubNameComparatorLevenshtein(boolean lowerCase) {
		super();
		this.lowerCase = lowerCase;
	}
	
	@Override
	public double compare(
			Club record1,
			Club record2,
			Correspondence<Attribute, Matchable> schemaCorrespondences) {
		if (this.lowerCase) return sim.calculate(record1.getName().toLowerCase(), record2.getName().toLowerCase());
		return sim.calculate(record1.getName(), record2.getName());
	}


}
