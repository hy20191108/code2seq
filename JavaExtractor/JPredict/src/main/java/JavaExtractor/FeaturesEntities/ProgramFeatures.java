package JavaExtractor.FeaturesEntities;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.stream.Collectors;

import com.github.javaparser.ast.Node;

public class ProgramFeatures {
    String target;

    ArrayList<ProgramRelation> paths = new ArrayList<>();
    String textContent;

    String filePath;

    public ProgramFeatures(String name, Path filePath, String textContent) {

        this.target = name;
        this.filePath = filePath.toAbsolutePath().toString();
        this.textContent = textContent;
    }

    @SuppressWarnings("StringBufferReplaceableByString")
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(target).append(" ");
        stringBuilder.append(paths.stream().map(ProgramRelation::toString).collect(Collectors.joining(" ")));

        return stringBuilder.toString();
    }

    public void addFeature(Property source, Node sourceNode, String path, Property target, Node targetNode) {
		ProgramRelation newRelation = new ProgramRelation(source, sourceNode, target, targetNode, path);
		paths.add(newRelation);
	}

    public boolean isEmpty() {
        return paths.isEmpty();
    }
}
