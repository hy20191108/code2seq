package JavaExtractor.FeaturesEntities;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.stream.Collectors;

import com.github.javaparser.ast.Node;

public class ProgramFeatures {
    String name;

    ArrayList<ProgramRelation> features = new ArrayList<>();
    String textContent;

    String filePath;

    public ProgramFeatures(String name, Path filePath, String textContent) {

        this.name = name;
        this.filePath = filePath.toAbsolutePath().toString();
        this.textContent = textContent;
    }

    @SuppressWarnings("StringBufferReplaceableByString")
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(name).append(" ");
        stringBuilder.append(features.stream().map(ProgramRelation::toString).collect(Collectors.joining(" ")));

        return stringBuilder.toString();
    }

    public void addFeature(Property source, Node sourceNode, String path, Property target, Node targetNode) {
		ProgramRelation newRelation = new ProgramRelation(source, sourceNode, target, targetNode, path);
		features.add(newRelation);
	}

    public boolean isEmpty() {
        return features.isEmpty();
    }
}
