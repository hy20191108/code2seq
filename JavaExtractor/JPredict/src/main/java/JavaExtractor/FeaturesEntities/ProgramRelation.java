package JavaExtractor.FeaturesEntities;

import com.github.javaparser.Position;
import com.github.javaparser.ast.Node;

public class ProgramRelation {
	private String name1;
	private String name2;
	private String shortPath;
	private String path;
	private Position name1Begin;
	private Position name1End;
	private Position name2Begin;
	private Position name2End;

	public ProgramRelation(Property sourceProp, Node sourceNode, Property targetProp, Node targetNode, String path) {
		// Text
		this.name1 = sourceProp.getName();
		this.name2 = targetProp.getName();
		this.shortPath = path;
		this.path = path;

		// Token Position
		this.name1Begin = sourceNode.getBegin();
		this.name1End = sourceNode.getEnd();
		this.name2Begin = targetNode.getBegin();
		this.name2End = targetNode.getEnd();

	}

	public String toString() {
		return String.format("%s,%s,%s", name1, shortPath, name2);
	}
}
