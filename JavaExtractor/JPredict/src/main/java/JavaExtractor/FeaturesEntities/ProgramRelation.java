package JavaExtractor.FeaturesEntities;

import com.github.javaparser.Position;
import com.github.javaparser.ast.Node;

public class ProgramRelation {
	transient Property sourceProp;
	transient Property targetProp;
	private String source;
	private String target;
	private String path;
	private Position sourceBegin;
	private Position sourceEnd;
	private Position targetBegin;
	private Position targetEnd;

	public ProgramRelation(Property sourceProp, Node sourceNode, Property targetProp, Node targetNode, String path) {
		this.sourceProp = sourceProp;
		this.targetProp = targetProp;
		
		this.source = sourceProp.getName();
		this.target = targetProp.getName();
		this.path = path;
		
		this.sourceBegin = sourceNode.getBegin();
		this.sourceEnd = sourceNode.getEnd();
		this.targetBegin = targetNode.getBegin();
		this.targetEnd = targetNode.getEnd();
		
	}

	public String toString() {
		return String.format("%s,%s,%s", sourceProp.getName(), path, targetProp.getName());
	}
}
